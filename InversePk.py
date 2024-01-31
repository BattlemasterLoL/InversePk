import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import ttk, StringVar, OptionMenu, filedialog


def matrix_eigen_window(
    a11,
    tau1,
    a12,
    a21,
    a22,
    tau2,
    a13,
    a23,
    a31,
    a32,
    a33,
    tau3,
    a14,
    a24,
    a34,
    a41,
    a42,
    a43,
    a44,
    tau4,
):
    global num_compartments
    window2 = tk.Toplevel()
    window2.title("Data")
    # mass transfer matrix
    s_matrix = np.array(
        [
            [a11, a12, a13, a14],
            [a21, a22, a23, a24],
            [a31, a32, a33, a34],
            [a41, a42, a43, a44],
        ]
    )
    s_adj = np.zeros((num_compartments, num_compartments))
    for i in range(num_compartments):
        for j in range(num_compartments):
            s_adj[i, j] = s_matrix[i, j]
    time_const = np.array([tau1, tau2, tau3, tau4])
    time_adj = np.zeros(num_compartments)
    for i in range(num_compartments):
        time_adj[i] = time_const[i]
    lam_matrix = np.diag(time_adj)
    m_matrix = np.dot(s_adj, np.dot(lam_matrix, np.linalg.inv(s_adj)))
    # calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(s_adj)
    # convert eigenvalues and eigenvectors to strings
    # eigenvalues_str = ', '.join(str(e) for e in eigenvalues)
    # eigenvectors_str = ', '.join(str(e) for e in eigenvectors)
    tk.Label(window2, text="Mass Transfer Matrix", font=("Arial", 16)).grid(
        row=0, column=0, padx=10, pady=10
    )
    tk.Label(window2, text=m_matrix[:num_compartments, :num_compartments]).grid(
        row=1, column=0, padx=10, pady=10
    )
    tk.Label(window2, text="Eigenvalues", font=("Arial", 16)).grid(
        row=2, column=0, padx=10, pady=10
    )
    tk.Label(window2, text=eigenvalues).grid(row=3, column=0, padx=10, pady=10)
    tk.Label(window2, text="Eigenvectors", font=("Arial", 16)).grid(
        row=4, column=0, padx=10, pady=10
    )
    tk.Label(window2, text=eigenvectors).grid(row=5, column=0, padx=10, pady=10)
    tk.Label(window2, text="Time Constants", font=("Arial", 16)).grid(
        row=6, column=0, padx=10, pady=10
    )
    tk.Label(window2, text=time_adj).grid(row=7, column=0, padx=10, pady=10)


def get_data():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    df = pd.read_csv(file_path, header=0)


def func1(t, a11=0, a12=0, a13=0, a14=0, tau1=0, tau2=0, tau3=0, tau4=0):
    # Check if any tau values are zero
    result = (
        a11 * (np.exp(-t / tau1) if tau1 != 0 else 0)
        + a12 * (np.exp(-t / tau2) if tau2 != 0 else 0)
        + a13 * (np.exp(-t / tau3) if tau3 != 0 else 0)
        + a14 * (np.exp(-t / tau4) if tau4 != 0 else 0)
    )
    return result


def func2(t, a21=0, a22=0, a23=0, a24=0, tau1=0, tau2=0, tau3=0, tau4=0):
    # Check if any tau values are zero
    result = (
        a21 * (np.exp(-t / tau1) if tau1 != 0 else 0)
        + a22 * (np.exp(-t / tau2) if tau2 != 0 else 0)
        + a23 * (np.exp(-t / tau3) if tau3 != 0 else 0)
        + a24 * (np.exp(-t / tau4) if tau4 != 0 else 0)
    )
    return result


def func3(t, a31=0, a32=0, a33=0, a34=0, tau1=0, tau2=0, tau3=0, tau4=0):
    # Check if any tau values are zero
    result = (
        a31 * (np.exp(-t / tau1) if tau1 != 0 else 0)
        + a32 * (np.exp(-t / tau2) if tau2 != 0 else 0)
        + a33 * (np.exp(-t / tau3) if tau3 != 0 else 0)
        + a34 * (np.exp(-t / tau4) if tau4 != 0 else 0)
    )
    return result


def func4(t, a41=0, a42=0, a43=0, a44=0, tau1=0, tau2=0, tau3=0, tau4=0):
    # Check if any tau values are zero
    result = (
        a41 * (np.exp(-t / tau1) if tau1 != 0 else 0)
        + a42 * (np.exp(-t / tau2) if tau2 != 0 else 0)
        + a43 * (np.exp(-t / tau3) if tau3 != 0 else 0)
        + a44 * (np.exp(-t / tau4) if tau4 != 0 else 0)
    )
    return result


def comboFunc(
    t,
    a11=0,
    tau1=0,
    a12=0,
    a21=0,
    a22=0,
    tau2=0,
    a13=0,
    a23=0,
    a31=0,
    a32=0,
    a33=0,
    tau3=0,
    a14=0,
    a24=0,
    a34=0,
    a41=0,
    a42=0,
    a43=0,
    a44=0,
    tau4=0,
):
    r1 = func1(
        t=t,
        a11=a11,
        a12=a12,
        a13=a13,
        a14=a14,
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
        tau4=tau4,
    )
    r2 = func2(
        t=t,
        a21=a21,
        a22=a22,
        a23=a23,
        a24=a24,
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
        tau4=tau4,
    )
    r3 = func3(
        t=t,
        a31=a31,
        a32=a32,
        a33=a33,
        a34=a34,
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
        tau4=tau4,
    )
    r4 = func4(
        t=t,
        a41=a41,
        a42=a42,
        a43=a43,
        a44=a44,
        tau1=tau1,
        tau2=tau2,
        tau3=tau3,
        tau4=tau4,
    )
    if num_compartments == 2:
        return np.concatenate((r1, r2))
    elif num_compartments == 3:
        return np.concatenate((r1, r2, r3))
    elif num_compartments == 4:
        return np.concatenate((r1, r2, r3, r4))
    else:
        pass


def run_simulation():
    global df
    global nCompartments
    global num_compartments

    num_compartments = int(nCompartments.get().split()[0])

    tcombo = df["Time"].values.ravel()
    column_names = [col for col in df.columns if "C" in col]
    ccombo = np.concatenate([df[col].values.ravel() for col in column_names])

    initial_params = np.ones((1, (3 * num_compartments) + 3))

    fittedparam, pcov = curve_fit(comboFunc, tcombo, ccombo, initial_params)
    zeros = np.zeros(20 - len(fittedparam))
    fittedparam_extended = np.concatenate((fittedparam, zeros))
    (
        a11,
        tau1,
        a12,
        a21,
        a22,
        tau2,
        a13,
        a23,
        a31,
        a32,
        a33,
        tau3,
        a14,
        a24,
        a34,
        a41,
        a42,
        a43,
        a44,
        tau4,
    ) = fittedparam_extended

    c1_fit = func1(df["Time"], a11, a12, a13, a14, tau1, tau2, tau3, tau4)
    c2_fit = func2(df["Time"], a21, a22, a23, a24, tau1, tau2, tau3, tau4)
    c3_fit = func3(df["Time"], a31, a32, a33, a34, tau1, tau2, tau3, tau4)
    c4_fit = func3(df["Time"], a41, a42, a43, a44, tau1, tau2, tau3, tau4)
    fitdf = pd.DataFrame({"C1": c1_fit, "C2": c2_fit, "C3": c3_fit, "C4": c4_fit})

    matrix_eigen_window(
        a11,
        tau1,
        a12,
        a21,
        a22,
        tau2,
        a13,
        a23,
        a31,
        a32,
        a33,
        tau3,
        a14,
        a24,
        a34,
        a41,
        a42,
        a43,
        a44,
        tau4,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmax = np.max(fitdf) * 1.2

    plt.figure(0)
    for i in range(num_compartments):
        plt.scatter(df.iloc[:, 0], df.iloc[:, i + 1], s=6, marker="o", label=f"C{i+1}")
    plt.title("Raw Data")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.show()

    plt.subplots(3, 1, sharey=True, figsize=(10, 8))
    for i in range(num_compartments):
        plt.subplot(num_compartments, 1, i + 1)
        plt.scatter(
            df.iloc[:, 0],
            df.iloc[:, i + 1],
            s=6,
            marker="o",
            label=f"Compartment {i+1}",
            color=colors[i % len(colors)],
        )
        plt.plot(df.iloc[:, 0], fitdf.iloc[:, i], color=colors[i % len(colors)])
        # plt.ylim([0, cmax])  # Set the same y-axis scale across all subplots
        plt.legend()
        plt.ylabel("Concentration")
    plt.suptitle("Fitted Data")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


# Initialize nessisary variables
df = None


root = tk.Tk()
root.title("Inverse Pk Solver")

frame_main = ttk.Frame(root, padding="10")
frame_main.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

settings_compartments = ttk.LabelFrame(frame_main, padding="10", text="Settings")
settings_compartments.grid(
    row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S)
)
nCompartments = StringVar()
nCompartments.set("# compartments")  # default value dropdown
comp_number = OptionMenu(
    settings_compartments,
    nCompartments,
    "2 Compartment",
    "3 Compartment",
    "4 Compartment",
)
comp_number.grid(row=0, column=0, padx=10, pady=10)
tk.Button(settings_compartments, text="Get Data", command=get_data).grid(
    row=1, column=0, padx=10, pady=10, ipadx=20
)

sym_run = tk.Button(root, text="Run Simulation", command=run_simulation)
sym_run.grid(row=5, column=0, columnspan=2, padx=10, pady=10, ipadx=20)


root.mainloop()
