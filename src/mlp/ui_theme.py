"""Paleta e tema dos gráficos, compartilhados pelos componentes da interface."""

PURPLE = "#8B5CF6"
CYAN = "#22D3EE"
MUTED = "#94A3B8"
LABELS = {"linear": "Linear", "sigmoid": "Sigmoid", "tanh": "Tanh", "relu": "ReLU"}


def chart_style(figure, height=350):
    """Tema comum dos gráficos Plotly com contraste e espaçamento consistentes."""
    figure.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="#172033", plot_bgcolor="#172033",
        font=dict(family="Inter, Arial, sans-serif", color="#F8FAFC", size=12),
        margin=dict(l=24, r=24, t=40, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#111827", font_color="#F8FAFC"),
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,0.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(148,163,184,0.08)", zeroline=False)
    return figure
