import openpyxl
import pytest

from enginetwin.epa import load_epa_map


def write_fuel_map(tmp_path, data_rows):
    """Same layout as EPA's 'Fuel Map Data.xlsx': title and citation, header row, unit row, then data."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fuel Map"
    ws.append(["2014 Mazda 2.0L SKYACTIV Engine LEV III Fuel - Fuel Map Data"])
    ws.append(["SUGGESTED CITATION:"])
    ws.append(["...citation text..."])
    ws.append([])
    ws.append(["Speed", "Torque", "Fuel Rate", "BSFC", "Efficiency"])
    ws.append(["RPM", "Nm", "g/sec", "g/kW*hr", "%"])
    for r in data_rows:
        ws.append(list(r))
    path = tmp_path / "fuel_map.xlsx"
    wb.save(path)
    return path


def test_epa_map_has_named_columns_and_efficiency_as_a_fraction(tmp_path):
    path = write_fuel_map(tmp_path, [(2000.0, 100.0, 1.9, 227.0, 36.0), (1000.0, 20.0, 0.4, 700.0, 12.0)])

    cells = load_epa_map(path)

    assert list(cells.columns) == ["rpm", "torque", "fuel_g_per_s", "bsfc_g_per_kwh", "efficiency"]
    assert cells["efficiency"].tolist() == pytest.approx([0.36, 0.12])
    assert cells["rpm"].tolist() == [2000.0, 1000.0]
