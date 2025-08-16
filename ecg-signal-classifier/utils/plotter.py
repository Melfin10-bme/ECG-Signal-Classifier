import plotly.graph_objs as go

def plot_ecg(time, ecg):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=ecg, mode="lines", name="ECG"))
    fig.update_layout(title="ECG Signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
    return fig
