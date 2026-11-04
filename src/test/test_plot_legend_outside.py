"""Leyenda fuera del área de dibujo y fondo oscuro de la página.

Cubre lo que se veía mal en los gráficos de backtest:

* ``_configure_legends`` saca la leyenda del marco con ``add_layout(leg, "above")``
  (horizontal, una sola fila) en vez de dejarla vertical dentro, tapando las
  primeras velas; la leyenda deja de estar en ``fig.center``.
* ``legend_placement="inside"`` recupera el comportamiento antiguo (útil en el
  gráfico en vivo, donde no hay franja que gastar) y ``"none"`` la apaga.
* ``output="file"`` escribe el HTML **sin** ``show()`` (no abre pestaña) y con la
  regla ``html,body{background:...}`` en el ``<head>``: sin ella el navegador
  pinta de blanco lo que el layout no cubre.
* ``PlotConfig.volume_alpha`` llega al glifo de volumen.
"""
import numpy as np
import pytest

pytest.importorskip("bokeh")


def _bt(n=400, seed=7):
    from tradetropy.core.data_types import KlineData
    from tradetropy.models.strategy import Strategy
    from tradetropy.backtest.engine import BacktestEngine

    rng = np.random.default_rng(seed)
    m = np.zeros((n, 7), dtype=np.float64)
    m[:, 0] = 1_700_000_000_000 + np.arange(n) * 60_000
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = np.r_[close[0], close[:-1]]
    m[:, 1] = open_
    m[:, 2] = np.maximum(open_, close) + 0.2
    m[:, 3] = np.minimum(open_, close) - 0.2
    m[:, 4] = close
    m[:, 5] = rng.uniform(1, 10, n)
    kl = KlineData(symbol="SYM", data=m, timeframe="1m", tick_size=0.01)

    class S(Strategy):
        def init(self):
            self.o = self.subscribe_ohlc("SYM", "1m", window_size=n)

        def on_data(self):
            pass

    bt = BacktestEngine.by_klines(S(), data=(kl,))
    bt.run()
    return bt


def _plot_and_capture(tmp_path, name="page.html", **cfg):
    """Grafica a fichero y devuelve ``(figura de precio, tema)``.

    Se engancha ``_build_layout``/``_show_layout`` de tradetropy para quedarse el
    layout ya montado con la leyenda configurada y, aun así, dejar que escriba el
    HTML de verdad. Mismo truco que usa el capturador de pitlane.
    """
    import tradetropy.plotting.plotting as P

    cap: dict = {}
    real_build, real_show = P._build_layout, P._show_layout

    def grab_build(figs):
        cap["figs"] = list(figs)
        return real_build(figs)

    def grab_show(layout, config, theme):
        cap["layout"], cap["theme"] = layout, theme
        real_show(layout, config, theme)

    P._build_layout, P._show_layout = grab_build, grab_show
    try:
        cfg.setdefault("output", "file")
        cfg.setdefault("filename", str(tmp_path / name))
        cfg.setdefault("plot_stats", False)
        _bt().plot(**cfg)
    finally:
        P._build_layout, P._show_layout = real_build, real_show

    figs = [f for f in cap["figs"] if hasattr(f, "x_range")]
    assert figs, "bt.plot() no devolvió figura de precio"
    # El panel de precio es el que lleva la banda de volumen colgada; el primero
    # de la lista puede ser el de retorno/equity, que tiene su propia leyenda.
    price = next((f for f in figs if "vol" in getattr(f, "extra_y_ranges", {})), figs[-1])
    return price, cap["theme"]


@pytest.mark.unit
class TestLegendPlacement:
    def test_default_moves_legend_out_of_the_frame(self, tmp_path):
        fig, _theme = _plot_and_capture(tmp_path, name="outside.html")

        assert fig.legend, "la figura no tiene leyenda"
        leg = fig.legend[0]
        assert leg.orientation == "horizontal"
        assert leg in list(fig.above), "la leyenda debe ocupar su franja arriba"
        assert leg not in list(fig.center), "no puede seguir dentro del área de velas"
        assert leg.spacing > 0, "las entradas no pueden ir pegadas"
        assert leg.click_policy == "hide", "el clic debe encender/apagar la capa"

    def test_inside_keeps_the_old_compact_legend(self, tmp_path):
        fig, _theme = _plot_and_capture(tmp_path, name="inside.html",
                                        legend_placement="inside")
        leg = fig.legend[0]
        assert leg.orientation == "vertical"
        assert leg.location == "top_left"
        assert leg in list(fig.center)
        assert leg not in list(fig.above)

    def test_none_hides_the_legend(self, tmp_path):
        fig, _theme = _plot_and_capture(tmp_path, name="none.html",
                                        legend_placement="none")
        assert fig.legend[0].visible is False


@pytest.mark.unit
class TestFileOutput:
    def test_file_output_does_not_open_a_browser(self, tmp_path, monkeypatch):
        import bokeh.plotting as bp

        def boom(*_a, **_k):  # pragma: no cover - sólo falla si se llama
            raise AssertionError("show() abrió el navegador")

        monkeypatch.setattr(bp, "show", boom)
        out = tmp_path / "no_tab.html"
        _bt().plot(output="file", filename=str(out), plot_stats=False)
        assert out.is_file() and out.stat().st_size > 0

    def test_page_background_is_painted_in_the_head(self, tmp_path):
        out = tmp_path / "bg.html"
        _bt().plot(output="file", filename=str(out), theme="dark", plot_stats=False)
        html = out.read_text(encoding="utf-8")
        head = html[: html.find("</head>")].replace(" ", "")
        assert "html,body{background:" in head, \
            "la regla html,body debe ir en el <head>, antes del primer pintado"


@pytest.mark.unit
class TestVolumeAlpha:
    def _volume_alphas(self, fig):
        from bokeh.models import VBar

        return [float(r.glyph.fill_alpha) for r in fig.renderers
                if isinstance(getattr(r, "glyph", None), VBar)
                and r.glyph.fill_alpha is not None]

    def test_config_alpha_reaches_the_volume_glyph(self, tmp_path):
        fig, _theme = _plot_and_capture(tmp_path, name="vol.html", volume_alpha=0.12)
        assert 0.12 in self._volume_alphas(fig), self._volume_alphas(fig)

    def test_theme_alpha_used_when_config_is_none(self, tmp_path):
        fig, theme = _plot_and_capture(tmp_path, name="vol_theme.html")
        assert float(theme["volume_alpha"]) in self._volume_alphas(fig), \
            self._volume_alphas(fig)
