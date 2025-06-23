import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------------------
# Helper functions
# -------------------------------

def d1(S, K, r, sigma, T):
    """
    Computes d1 used in Black-Scholes.
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, r, sigma, T):
    """
    Computes d2 used in Black-Scholes.
    """
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

def call_price(S, K, r, sigma, T):
    """
    Black-Scholes price for a call option.
    """
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)

def put_price(S, K, r, sigma, T):
    """
    Black-Scholes price for a put option.
    """
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

def greeks(S, K, r, sigma, T, option_type="call"):
    """
    Return the main Greeks for Black-Scholes.

    Parameters
    ----------
    S : float or numpy array
        Underlying price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity (in years).
    option_type : str
        'call' or 'put'.
    """
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    pdf_d1 = norm.pdf(d_1)
    cdf_d1 = norm.cdf(d_1)
    cdf_d2 = norm.cdf(d_2)
    
    if option_type == "call":
        # Delta
        delta = cdf_d1
        # Gamma (same for call & put)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        # Vega (same for call & put, but typically scaled by 0.01 if desired in %)
        vega = S * pdf_d1 * np.sqrt(T)
        # Theta
        theta = - (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) \
                - r * K * np.exp(-r * T) * cdf_d2
        # Rho
        rho = K * T * np.exp(-r * T) * cdf_d2
    else:  # put
        # Delta
        delta = cdf_d1 - 1
        # Gamma (same for call & put)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        # Vega (same for call & put)
        vega = S * pdf_d1 * np.sqrt(T)
        # Theta
        theta = - (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) \
                + r * K * np.exp(-r * T) * norm.cdf(-d_2)
        # Rho
        rho = - K * T * np.exp(-r * T) * norm.cdf(-d_2)

    return delta, gamma, vega, theta, rho

# -------------------------------
# Streamlit app
# -------------------------------

def main():
    # Top control area
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_type = st.selectbox("Option type:", ("call", "put"))
            K = st.slider("Strike price:", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
        
        with col2:
            r = st.slider("Risk-free rate:", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
            sigma = st.slider("Volatility:", min_value=0.05, max_value=1.0, value=0.2, step=0.01)
        
        with col3:
            T = st.slider("Time to maturity:", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
        

    # Underlying price range
    S_values = np.linspace(1, 2*K, 200)

    # Compute Greeks
    delta_vals, gamma_vals, vega_vals, theta_vals, rho_vals = greeks(S_values, K, r, sigma, T, option_type)

    # Option Price (just for reference, can show on the subplot)
    if option_type == "call":
        price_vals = call_price(S_values, K, r, sigma, T)
    else:
        price_vals = put_price(S_values, K, r, sigma, T)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0, 2]
    ax4 = axes[1, 0]
    ax5 = axes[1, 1]
    ax6 = axes[1, 2]

    # Price
    ax1.plot(S_values, price_vals, label=f"{option_type.capitalize()} Price", color='blue')
    ax1.set_title(f"{option_type.capitalize()} Price")
    ax1.set_xlabel("Underlying Price (S)")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    # Delta
    ax2.plot(S_values, delta_vals, label="Delta", color='green')
    ax2.set_title("Delta")
    ax2.set_xlabel("S")
    ax2.set_ylabel("Delta")
    ax2.grid(True)

    # Gamma
    ax3.plot(S_values, gamma_vals, label="Gamma", color='red')
    ax3.set_title("Gamma")
    ax3.set_xlabel("S")
    ax3.set_ylabel("Gamma")
    ax3.grid(True)

    # Vega
    ax4.plot(S_values, vega_vals, label="Vega", color='purple')
    ax4.set_title("Vega")
    ax4.set_xlabel("S")
    ax4.set_ylabel("Vega")
    ax4.grid(True)

    # Theta
    ax5.plot(S_values, theta_vals, label="Theta", color='brown')
    ax5.set_title("Theta")
    ax5.set_xlabel("S")
    ax5.set_ylabel("Theta")
    ax5.grid(True)

    # Rho
    ax6.plot(S_values, rho_vals, label="Rho", color='orange')
    ax6.set_title("Rho")
    ax6.set_xlabel("S")
    ax6.set_ylabel("Rho")
    ax6.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
