import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Judul aplikasi dengan huruf tebal
st.markdown("<h1 style='font-weight:bold;'>Statistika Regresi App</h1>", unsafe_allow_html=True)

st.write("Masukkan nilai X dan Y (Setiap nilai dipisahkan baris baru, tekan Enter setelah setiap angka).")

# Input nilai X
x_input = st.text_area("Masukkan nilai X (satu per baris):", height=180)
# Input nilai Y
y_input = st.text_area("Masukkan nilai Y (satu per baris):", height=180)

# Fungsi untuk parsing input menjadi list of float
def parse_input(input_str):
    return [float(i) for i in input_str.strip().split('\n') if i.strip() != ""]

# Tombol proses
if st.button("Hitung Data!"):
    try:
        x_list = parse_input(x_input)
        y_list = parse_input(y_input)
        
        if len(x_list) != len(y_list):
            st.error("Jumlah nilai X dan Y harus sama!")
        elif len(x_list) < 2:
            st.error("Masukkan minimal 2 data untuk X dan Y!")
        else:
            X = np.array(x_list).reshape(-1, 1)
            Y = np.array(y_list)
            
            # Garis Regresi
            model = LinearRegression()
            model.fit(X, Y)
            a = model.intercept_
            b = model.coef_[0]
            
            # Persamaan regresi
            st.subheader("Persamaan Garis Regresi")
            st.write("(4 angka di belakang koma).")
            st.latex(f"\\hat{{y}} = {b:.4f}x + {a:.4f}")
            
            # Koefisien korelasi
            r = np.corrcoef(x_list, y_list)[0, 1]
            st.subheader("Koefisien Korelasi (r)")
            st.write(f"r = {r:.4f}")
            
            # Koefisien determinasi
            r2 = model.score(X, Y)
            st.subheader("Koefisien Determinasi (r²)")
            st.write(f"r² = {r2:.4f} ({r2*100:.2f}%)")
            
            # Plot kurva regresi
            st.subheader("Kurva Regresi Linear")
        plt.figure(figsize=(8,5))
        plt.scatter(x_list, y_list, color='blue', label='Data')
        plt.plot(x_list, model.predict(X), color='green', label='Regresi Linear')
        plt.xlabel('X (Variabel Independen)')
        plt.ylabel('Y (Variabel Dependen)')
        plt.title('Kurva Regresi Linear')
        plt.legend()
        st.pyplot(plt)
        st.write("KAHAYA AYU MAHARENI (17 / XI.G)")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")