
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try optional PyTorch for the neural net optimizer; fall back gracefully if unavailable
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

st.set_page_config(page_title="FX Hedging Simulator (EUR/PLN)", layout="wide")

st.title("FX Hedging Simulator (EUR/PLN)")
st.write(
    """
    This app simulates PLN cash flows converted into a **December dividend in EUR** under FX risk.
    - You provide **annual FX volatility**, **current spot**, and **monthly interest rates** for EUR and PLN (maturities 1–12 months).
    - We convert annual vol to monthly by dividing by √12 and simulate a **monthly random walk (GBM, zero drift)**.
    - You enter **12 monthly profits in PLN (Jan–Dec)**.
    - A **hedged fraction** of each month's profit is converted at the **CIP-implied forward for that maturity**, then **invested in EUR** to December.
      The **unhedged fraction** is converted at **December spot**.
    - The app shows the **expected December dividend (EUR)** and its **volatility** across simulations.
    - Optionally, a **small neural network** (if PyTorch is available) searches for month-by-month hedge weights that trade off mean vs. volatility to sketch an efficient frontier.
    
    **Quote convention:** We use **EUR per PLN** (EUR/PLN). One PLN is worth *S* euros.
    **Covered Interest Parity (CIP):** For maturity *m* months, forward is
    \( F_m = S_0 \times \frac{\prod_{k=1}^m (1+r^{\text{EUR}}_k)}{\prod_{k=1}^m (1+r^{\text{PLN}}_k)} \).
    """
)

with st.sidebar:
    st.header("Inputs")
    seed = st.number_input("Random seed", value=42, step=1)
    n_paths = st.number_input("Simulation paths", value=5000, min_value=100, step=100)
    annual_vol = st.number_input("Annual FX volatility (as decimal, e.g. 0.12 for 12%)", value=0.12, min_value=0.0, format="%.4f")
    spot = st.number_input("Current spot S₀ (EUR per PLN)", value=0.23, min_value=0.0, format="%.6f")
    st.caption("Typical market convention is PLN per EUR; here we use **EUR per PLN** to match the academic convention in this app.")
    
    st.subheader("Monthly EUR interest rates r^EUR_k")
    eur_rates = []
    for k in range(1, 13):
        eur_rates.append(st.number_input(f"EUR r{k} (month {k})", value=0.003 if k <= 6 else 0.004, step=0.001, format="%.4f", key=f"eur{k}"))
    eur_rates = np.array(eur_rates, dtype=float)

    st.subheader("Monthly PLN interest rates r^PLN_k")
    pln_rates = []
    for k in range(1, 13):
        pln_rates.append(st.number_input(f"PLN r{k} (month {k})", value=0.006 if k <= 6 else 0.007, step=0.001, format="%.4f", key=f"pln{k}"))
    pln_rates = np.array(pln_rates, dtype=float)

    st.subheader("Monthly profits in PLN (Jan–Dec)")
    profits = []
    for k, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1):
        profits.append(st.number_input(f"{m} profit (PLN)", value=1_000_000.0, step=10000.0, format="%.2f", key=f"p{k}"))
    profits = np.array(profits, dtype=float)

    st.subheader("Hedging control")
    hedge_frac = st.slider("Uniform hedged fraction (applied to each month)", 0.0, 1.0, 0.5, 0.01)
    use_nn = st.checkbox("Use neural network to search month-by-month hedge weights (optional)", value=False, help="Requires PyTorch. Traces a mean–volatility frontier by varying risk aversion.")

# Derived quantities
np.random.seed(seed)
if TORCH_AVAILABLE:
    torch.manual_seed(seed)

monthly_vol = annual_vol / np.sqrt(12.0)
st.write(f"**Monthly volatility** = {monthly_vol:.6f}")

# Compute term structures (cumulative factors) and forwards by CIP
eur_acc = np.cumprod(1.0 + eur_rates)
pln_acc = np.cumprod(1.0 + pln_rates)
forwards = spot * (eur_acc / pln_acc)  # F_m for m = 1..12 (EUR per PLN)

f_df = pd.DataFrame({
    "maturity_month": np.arange(1, 13),
    "EUR_factor": eur_acc,
    "PLN_factor": pln_acc,
    "Forward_EUR_per_PLN": forwards
})
with st.expander("Show CIP forward curve table"):
    st.dataframe(f_df.style.format({"EUR_factor": "{:.6f}", "PLN_factor": "{:.6f}", "Forward_EUR_per_PLN": "{:.6f}"}), use_container_width=True)

# Simulate monthly spot path via GBM with zero drift under real-world measure
# S_t = S_{t-1} * exp( -0.5*sigma^2 + sigma * eps )
def simulate_spot_paths(S0, sigma_m, T=12, n=n_paths):
    eps = np.random.randn(n, T)
    increments = (-0.5 * sigma_m**2) + sigma_m * eps
    logS = np.cumsum(increments, axis=1) + np.log(S0)
    S = np.exp(logS)
    return S  # shape (n, 12)

S_paths = simulate_spot_paths(spot, monthly_vol, 12, int(n_paths))
S_dec = S_paths[:, -1]  # December spot per path (EUR per PLN)

# Compute December dividend in EUR for a given vector of monthly hedge weights h[1..12] in [0,1]
# Hedged portion for month m uses forward F_m at t=0 for maturity m, delivers EUR in month m,
# then grows in EUR at rates r^{EUR}_{m+1..12} to December.
# Unhedged portion converts at S_Dec in December.
eur_grow_from_m = np.ones(12)
# growth from month m to Dec (exclude month m, i.e., multiply months m+1..12)
for m in range(12):
    if m < 11:
        eur_grow_from_m[m] = np.prod(1.0 + eur_rates[m+1:])
    else:
        eur_grow_from_m[m] = 1.0

profits = profits.astype(float)
forwards = forwards.astype(float)

def december_dividend_eur(S_dec_vec, h_vec):
    """
    S_dec_vec: shape (n_paths,)
    h_vec: shape (12,), values in [0,1]
    Returns: array shape (n_paths,) of December EUR dividends
    """
    h = np.clip(h_vec, 0.0, 1.0)
    # Hedged leg: sum over months of h_m * P_m * F_m * EUR_growth_to_Dec
    hedged_eur = np.sum(h * profits * forwards * eur_grow_from_m)
    # Unhedged leg: depends on December spot (pathwise): sum over months of (1-h_m)*P_m * S_Dec
    unhedged_pln = np.sum((1.0 - h) * profits)  # PLN amount left to convert in December
    unhedged_eur_paths = unhedged_pln * S_dec_vec  # pathwise
    return hedged_eur + unhedged_eur_paths

# Baseline: uniform hedge fraction
h_uniform = np.full(12, hedge_frac)
divs_uniform = december_dividend_eur(S_dec, h_uniform)
mean_uniform = float(np.mean(divs_uniform))
std_uniform = float(np.std(divs_uniform, ddof=1))

col1, col2 = st.columns(2)
with col1:
    st.metric("Expected December dividend (EUR)", f"{mean_uniform:,.2f}")
with col2:
    st.metric("Dividend volatility (EUR std dev)", f"{std_uniform:,.2f}")

# Build frontier: sweep uniform h in [0,1]
sweep_points = []
for h in np.linspace(0.0, 1.0, 26):
    divs = december_dividend_eur(S_dec, np.full(12, h))
    sweep_points.append((np.mean(divs), np.std(divs, ddof=1), h))
sweep_df = pd.DataFrame(sweep_points, columns=["mean_eur", "std_eur", "uniform_h"])

# Optional: Neural-net search for month-by-month hedge weights to trace a frontier
nn_points = []
if use_nn:
    if not TORCH_AVAILABLE:
        st.warning("PyTorch not found. Install PyTorch to enable the neural network optimizer, or uncheck the option.")
    else:
        st.info("Running neural network optimizer... this may take ~10–30 seconds.")
        # Fixed S_dec sample for the NN (so it's optimizing on the same draw)
        S_dec_torch = torch.tensor(S_dec, dtype=torch.float32)

        # Small network: input is a learned constant, output 12 hedge weights in (0,1) via sigmoid
        class HedgeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(16))
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 12)
                self.act = nn.ReLU()
                self.sig = nn.Sigmoid()
            def forward(self):
                x = self.act(self.fc1(self.w))
                h = self.sig(self.fc2(x))  # 12 weights in (0,1)
                return h

        # Precompute tensors
        profits_t = torch.tensor(profits, dtype=torch.float32)
        forwards_t = torch.tensor(forwards, dtype=torch.float32)
        eur_grow_t = torch.tensor(eur_grow_from_m, dtype=torch.float32)

        def dec_dividend_torch(S_dec_t, h_t):
            hedged_eur = torch.sum(h_t * profits_t * forwards_t * eur_grow_t)
            unhedged_pln = torch.sum((1.0 - h_t) * profits_t)
            unhedged_eur = unhedged_pln * S_dec_t  # vector over paths
            return hedged_eur + unhedged_eur  # vector over paths

        # Optimize for multiple lambdas to trace a curve: maximize mean - λ * std
        lambdas = np.linspace(0.0, 5.0, 16)
        for lam in lambdas:
            net = HedgeNet()
            opt = torch.optim.Adam(net.parameters(), lr=0.03)
            for step in range(800):
                opt.zero_grad()
                h_out = net()  # shape (12,)
                divs_vec = dec_dividend_torch(S_dec_torch, h_out)  # shape (n_paths,)
                mean = torch.mean(divs_vec)
                std = torch.std(divs_vec, unbiased=True)
                loss = -(mean - float(lam) * std)
                loss.backward()
                opt.step()
            with torch.no_grad():
                h_best = net().detach().numpy()
                divs_best = dec_dividend_torch(S_dec_torch, torch.tensor(h_best)).detach().numpy()
                nn_points.append((float(np.mean(divs_best)), float(np.std(divs_best, ddof=1)), h_best))

        nn_df = pd.DataFrame([(m,s) for (m,s,_) in nn_points], columns=["mean_eur","std_eur"])
    # end if torch available

# Plot mean vs std
fig, ax = plt.subplots()
ax.scatter(sweep_df["std_eur"], sweep_df["mean_eur"], label="Uniform h sweep", s=30)
if use_nn and TORCH_AVAILABLE and len(nn_points) > 0:
    ax.scatter(nn_df["std_eur"], nn_df["mean_eur"], marker="x", label="NN-optimized h (frontier)", s=40)
ax.scatter([std_uniform], [mean_uniform], marker="D", s=60, label=f"Selected h={hedge_frac:.2f}")
ax.set_xlabel("Dividend volatility (EUR std dev)")
ax.set_ylabel("Expected December dividend (EUR)")
ax.set_title("Mean–Volatility Tradeoff")
ax.legend()
st.pyplot(fig)

# Results table and download
out_rows = []
out_rows.append({
    "strategy": f"uniform_h_{hedge_frac:.2f}",
    "mean_eur": mean_uniform,
    "std_eur": std_uniform
})
for _, row in sweep_df.iterrows():
    out_rows.append({"strategy": f"uniform_h_{row['uniform_h']:.2f}", "mean_eur": row["mean_eur"], "std_eur": row["std_eur"]})
if use_nn and TORCH_AVAILABLE and len(nn_points) > 0:
    for i, (m,s,hvec) in enumerate(nn_points):
        out_rows.append({"strategy": f"nn_opt_{i:02d}", "mean_eur": m, "std_eur": s})

results_df = pd.DataFrame(out_rows)
st.download_button("Download summary CSV", data=results_df.to_csv(index=False).encode("utf-8"), file_name="hedging_summary.csv", mime="text/csv")

st.markdown("---")
st.subheader("Notes & Assumptions")
st.markdown(
    """
    - **Profits** are provided in PLN and assumed realized in the labeled month.
    - The **hedged fraction** of each month’s profit is locked via a **forward entered at t=0** with the corresponding maturity **(1–12 months)** and then **invested in EUR to December** using the EUR monthly rates you provided.
    - The **unhedged fraction** is converted at **December spot**.
    - FX dynamics: geometric Brownian motion (zero drift) with monthly volatility = annual volatility / √12.
    - Interest rates are **simple monthly rates** (not annualized). If you have annualized rates, convert them to monthly first.
    - Quote: **EUR per PLN (EUR/PLN)** throughout.
    """
)
