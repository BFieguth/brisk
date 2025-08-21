from pathlib import Path

import pytest
from unittest import mock
import matplotlib.pyplot as plt
import plotnine as pn
import pandas as pd

from brisk.services.io import IOService

@pytest.fixture
def io_service(tmp_path):
    return IOService(
        name="io",
        results_dir=Path(tmp_path / "test" / "path"),
        output_dir=Path(tmp_path / "test" / "path"/ "results")
    )


class TestIOService:
    def test_init(self, io_service, tmp_path):
        assert io_service.name == "io"
        assert io_service.results_dir == Path(tmp_path / "test" / "path")
        assert io_service.output_dir == Path(tmp_path / "test" / "path"/ "results")

    def test_set_output_dir(self, io_service, tmp_path):
        io_service.set_output_dir(
            Path(tmp_path / "test" / "path" / "results2")
        )
        assert io_service.output_dir == Path(tmp_path / "test" / "path" / "results2")

    @mock.patch("json.dump")
    def test_save_to_json(self, mock_json_dump, io_service):
        io_service._other_services["reporting"] = mock.MagicMock()

        results = {"results": "1.234"}
        output_path = io_service.output_dir / "results.json"
        metadata = {"created_at": "2025-01-01"}

        io_service.save_to_json(results, output_path, metadata)

        mock_json_dump.assert_called_once()
        assert io_service._other_services["reporting"].store_table_data.call_count == 1

    def test_save_plot_pyplot_stack(self, io_service):
        io_service._other_services["reporting"] = mock.MagicMock()
                
        plt.scatter(range(10), range(10))
        plt.title("Test Plot")
        output_path = io_service.output_dir / "plot.png"
        metadata = {"created_at": "2025-01-01"}

        io_service.save_plot(output_path, metadata)

        assert output_path.exists()
        assert io_service._other_services["reporting"].store_plot_svg.call_count == 1
        assert plt.get_fignums() == []

    def test_save_plot_object(self, io_service):
        io_service._other_services["reporting"] = mock.MagicMock()

        plot = pn.ggplot(
            data=pd.DataFrame({"x": range(10), "y": range(10)}),
            mapping=pn.aes(x="x", y="y")
        ) + pn.geom_point() + pn.labs(title="Test Plot")

        output_path = io_service.output_dir / "plot.png"
        metadata = {"created_at": "2025-01-01"}

        io_service.save_plot(output_path, metadata, plot)

        assert output_path.exists()
        assert io_service._other_services["reporting"].store_plot_svg.call_count == 1
