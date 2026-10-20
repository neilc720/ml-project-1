"""Data side of the final model: CSV loading, BRFSS sentinel recoding and the two feature maps.

* ``load_data`` parses the three course CSVs once and caches them as ``.npy`` files.
* ``SENTINELS`` maps each survey column that has numeric codebook codes to its "don't know /
  refused / not asked" codes (recoded to missing) and "none" codes (recoded to zero), from the CDC
  2015 BRFSS codebook; other columns are not recoded. For ``MAXVO2_``/``FC60_`` and
  ``PAFREQ1_``/``PAFREQ2_``/``STRFREQ_`` this CSV stores the missing code with the codebook's implied
  decimals applied (999 and 99), so both spellings are listed.
* ``TreeBinner`` (tree models) drops the identifier columns and any all-missing or constant column,
  recodes sentinels and bins each column into at most ``max_bins`` quantile bins, with bin 0
  reserved for missing values.
* ``WideFeatureMap`` (logistic model) builds one-hot levels with "other"/"missing", clipped and
  standardised numerics and quantile-bin indicators, in row chunks.

All statistics are fitted on the rows passed to ``fit``. NumPy and the standard library only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

MISSING_BIN = 0

# Interview, phone and survey-design columns removed before modelling (``_STATE`` is kept).
IDENTIFIER_COLUMNS: tuple[str, ...] = tuple(
    (
        "FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU CTELENUM PVTRESD1 COLGHOUS STATERES "
        "CELLFON3 LADULT CTELNUM1 CELLFON2 CADULT PVTRESD2 CCLGHOUS CSTATE LANDLINE QSTVER QSTLANG "
        "_STSTR _STRWT _RAWRAKE _WT2RAKE _DUALUSE _DUALCOR _LLCPWT"
    ).split()
)

# (missing codes, zero codes) -> columns; columns not listed have no numeric sentinel.
_SENTINEL_GROUPS: dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]] = {
    ((7, 9), ()): (
        "LANDLINE GENHLTH HLTHPLN1 PERSDOC2 MEDCOST CHECKUP1 BPHIGH4 BPMEDS BLOODCHO CHOLCHK TOLDHI2 "
        "CVDSTRK3 ASTHMA3 ASTHNOW CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY DIABETE3 "
        "RENTHOM1 NUMHHOL2 NUMPHON2 CPDEMO1 VETERAN3 INTERNET PREGNANT QLACTLM2 USEEQUIP BLIND DECIDE "
        "DIFFWALK DIFFDRES DIFFALON SMOKE100 SMOKDAY2 STOPSMK2 USENOW3 EXERANY2 LMTJOIN3 ARTHDIS2 "
        "ARTHSOCL SEATBELT FLUSHOT6 PNEUVAC3 HIVTST6 PDIABTST PREDIAB1 EYEEXAM DIABEYE DIABEDU CAREGIV1 "
        "CRGVLNG1 CRGVHRS1 CRGVPERS CRGVHOUS CRGVMST2 CRGVEXPT VIPRFVS2 VIEYEXM2 VIINSUR2 CIMEMLOS "
        "CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS WTCHSALT DRADVISE ASYMPTOM ASTHMED3 ASINHALR HAREHAB1 "
        "STREHAB1 CVDASPRN ASPUNSAF RLIVPAIN RDUCHART RDUCSTRK ARTTODAY ARTHWGT ARTHEXER ARTHEDU TETANUS "
        "HPVADVC2 SHINGLE2 HADMAM HOWLONG HADPAP2 LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM LENGEXAM "
        "BLDSTOOL LSTBLDS3 HADSIGM3 HADSGCO1 LASTSIG3 PCPSAAD2 PCPSADI1 PCPSARE1 PSATEST1 PSATIME "
        "PCPSARS1 PCDMDECN SCNTMNY1 SCNTMEL1 SCNTPAID SCNTLPAD SXORIENT TRNSGNDR RCSRLTN2 CASTHDX2 "
        "CASTHNO2 EMTSUPRT LSATISFY MISTMNT ADANXEV DRNKANY5"
    ).split(),
    ((9,), ()): (
        "MARITAL EDUCA EMPLOY1 INSULIN PCPSADE1 RCSGENDR _CHISPNC _RFHLTH _HCVU651 _RFHYPE5 _CHOLCHK "
        "_RFCHOL _LTASTH1 _CASTHM1 _ASTHMS1 _HISPANC _RACE _RACEG21 _RACEGR3 _RFBMI5 _CHLDCNT _EDUCAG "
        "_INCOMG _SMOKER3 _RFSMOK3 _RFBING5 _RFDRHV5 _FRTLT1 _VEGLT1 _TOTINDA PAMISS1_ _PACAT1 _PAINDX1 "
        "_PA150R2 _PA300R2 _PA30021 _PASTRNG _PAREC1 _PASTAE1 _LMTACT1 _LMTWRK1 _LMTSCL1 _RFSEAT2 "
        "_RFSEAT3 _FLSHOT6 _PNEUMO2 _AIDTST3"
    ).split(),
    ((77, 99), ()): (
        "HHADULT INCOME2 LASTSMK2 AVEDRNK2 MAXDRNKS EXRACT11 EXRACT21 JOINPAIN IMFVPLAC WHRTST10 "
        "CRGVREL1 CRGVPRB1 VINOCRE2 HPVADSHT _CRACE1 _CPRACE _PRACE1 _MRACE1"
    ).split(),
    ((77, 99), (88,)): (
        "PHYSHLTH MENTHLTH POORHLTH DRNK3GE5 DOCTDIAB CHKHEMO3 FEETCHK ADPLEASR ADDOWN ADSLEEP ADENERGY "
        "ADEAT1 ADFAIL ADTHINK ADMOVE"
    ).split(),
    ((7,), ()): (
        "VIDFCLT2 VIREDIF3 VICTRCT4 VIGLUMA2 VIMACDG2 ASATTACK"
    ).split(),
    ((777, 999), (555,)): (
        "FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1"
    ).split(),
    ((777, 999), ()): (
        "EXEROFT1 EXERHMM1 EXEROFT2 EXERHMM2 LONGWTCH"
    ).split(),
    ((777, 999), (888,)): (
        "ALCDAY5 STRENGTH BLDSUGAR FEETCHK2 ASACTLIM"
    ).split(),
    ((99, 99000), ()): (
        "PAFREQ1_ PAFREQ2_ STRFREQ_"
    ).split(),
    ((7777, 9999), ()): (
        "WEIGHT2 HEIGHT3"
    ).split(),
    ((777777, 999999), ()): (
        "FLSHTMY2 HIVTSTD3"
    ).split(),
    ((97, 99), (98,)): (
        "SCNTWRK1 SCNTLWK1"
    ).split(),
    ((98, 99), ()): (
        "DIABAGE2 ASTHMAGE"
    ).split(),
    ((98,), (88,)): (
        "ASERVIST ASDRVIST"
    ).split(),
    ((999, 99900), ()): (
        "MAXVO2_ FC60_"
    ).split(),
    ((14,), ()): (
        "_AGEG5YR"
    ).split(),
    ((3,), ()): (
        "_AGE65YR"
    ).split(),
    ((7,), (8,)): (
        "ASNOSLEP"
    ).split(),
    ((900,), ()): (
        "DROCDY3_"
    ).split(),
    ((98, 99), (88,)): (
        "ASRCHKUP"
    ).split(),
    ((99,), (88,)): (
        "CHILDREN"
    ).split(),
    ((99900,), ()): (
        "_DRNKWEK"
    ).split(),
    ((99999,), ()): (
        "WTKG3"
    ).split(),
}
SENTINELS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    name: codes for codes, names in _SENTINEL_GROUPS.items() for name in names
}


def schema_for_names(feature_names: Sequence[str]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """(missing codes, zero codes) for each name, in input order; unknown names get no recoding."""
    return [SENTINELS.get(name, ((), ())) for name in feature_names]


def file_hash(path: str | Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _csv_lines(source: Path) -> Iterator[str]:
    """Data lines of a course CSV with every empty field written as ``nan`` (header skipped)."""
    with source.open() as stream:
        next(stream)
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if ", " in line or ",\t" in line:
                line = ",".join(field.strip() or "nan" for field in line.split(","))
            # Two passes cover adjacent empty fields without a parser dependency.
            line = line.replace(",,", ",nan,").replace(",,", ",nan,")
            if line.endswith(","):
                line += "nan"
            if line.startswith(","):
                line = "nan" + line
            yield line


def load_data(data_dir: str | Path, cache_dir: str | Path, include_test: bool = False) -> dict[str, Any]:
    """Parse the course CSVs (cached as ``.npy`` keyed by SHA-256) and check their IDs and labels.

    Returns ``x`` (rows x 321 features, NaN for blanks), ``y`` (0/1), ``ids``, ``names`` and the
    source ``hashes``; with ``include_test`` also ``x_test`` and ``test_ids``.
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    files = ["x_train", "y_train"] + (["x_test"] if include_test else [])
    headers, hashes = {}, {}
    for stem in files:
        source = data_dir / (stem + ".csv")
        digest = file_hash(source)
        hashes[stem] = digest
        with source.open() as stream:
            headers[stem] = [field.strip() for field in stream.readline().strip().split(",")]
        if (len(headers[stem]) < 2 or not all(headers[stem])
                or len(set(headers[stem])) != len(headers[stem])):
            raise ValueError(f"{source.name} must have a nonempty, unique ID/column header")
        if stem == "y_train" and len(headers[stem]) != 2:
            raise ValueError("y_train.csv must contain exactly an ID and a label column")
        target = cache_dir / (stem + ".npy")
        marker = target.with_suffix(".sha256")
        if not target.exists() or not marker.exists() or marker.read_text() != digest:
            data_lines = _csv_lines(source)
            first_line = next(data_lines, None)
            if first_line is None:
                raise ValueError(f"{source.name} contains no data rows")
            matrix = np.loadtxt(itertools.chain([first_line], data_lines), delimiter=",", ndmin=2)
            if matrix.shape[1] != len(headers[stem]):
                raise ValueError(f"{source.name} header and data column counts differ")
            np.save(target, matrix)
            marker.write_text(digest)
        arrays[stem] = np.load(target, mmap_mode="r")
        matrix = arrays[stem]
        if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != len(headers[stem]):
            raise ValueError(f"{source.name} must contain a nonempty matrix matching its header")
        ids = matrix[:, 0]
        if (not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all()
                or np.any(ids < np.iinfo(np.int64).min)
                or np.any(ids >= float(np.iinfo(np.int64).max))):
            raise ValueError(f"{source.name} IDs must be finite integers in int64 range")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs in {source.name}")
    x, target = arrays["x_train"], arrays["y_train"]
    if not np.array_equal(x[:, 0], target[:, 0]):
        raise ValueError("Feature and label IDs are not aligned")
    if not np.isin(target[:, 1], [-1, 1]).all():
        raise ValueError("Expected -1/+1 source labels")
    result = dict(x=x[:, 1:], y=(target[:, 1] == 1).astype(float),
                  ids=x[:, 0].astype(np.int64), names=headers["x_train"][1:], hashes=hashes)
    if include_test:
        if headers["x_test"] != headers["x_train"]:
            raise ValueError("Train/test feature headers differ")
        result.update(x_test=arrays["x_test"][:, 1:],
                      test_ids=arrays["x_test"][:, 0].astype(np.int64))
    return result


@dataclass(frozen=True)
class TreeFeatureConfig:
    """Options of the tree preset.

    Attributes:
        drop_columns: Columns removed by name (administrative identifiers).
        max_bins: Maximum number of non-missing bins per column (2..255).
        sample_size: Rows sampled for quantile cut points of high-cardinality columns.
        seed: Seed of the quantile sampling.
        chunk_rows: Rows transformed per block to bound temporary memory.
    """

    drop_columns: tuple[str, ...] = IDENTIFIER_COLUMNS
    max_bins: int = 64
    sample_size: int = 100_000
    seed: int = 2026
    chunk_rows: int = 65_536

    def __post_init__(self) -> None:
        if not 2 <= int(self.max_bins) <= 255:
            raise ValueError("max_bins must be between 2 and 255")
        if self.sample_size < 1 or self.chunk_rows < 1:
            raise ValueError("sample_size and chunk_rows must be positive")


def _validate_matrix(x: Any) -> np.ndarray:
    """Return ``x`` as a 2-D float array, allowing NaN but not infinities."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional feature matrix")
    if np.isinf(x).any():
        raise ValueError("x must not contain infinite values")
    return x


def _clean_column(values: np.ndarray, missing: Sequence[int], zero: Sequence[int]) -> np.ndarray:
    """Copy one column, mapping 'none' codes to 0 and sentinel codes to NaN."""
    col = np.array(values, dtype=float, copy=True)
    for code in zero:
        col[col == code] = 0.0
    for code in missing:
        col[col == code] = np.nan
    return col


class TreeBinner:
    """Select columns, recode codebook sentinels and bin values to ``uint8``.

    After ``fit`` the attributes ``column_indices_`` (kept input columns), ``cuts_``
    (one sorted array of cut points per kept column) and ``n_bins_`` (bins per
    column, including the missing bin) are set.
    ``transform`` maps a value to ``1 + #cuts <= value``; NaN maps to 0.
    """

    def __init__(self, feature_names: Sequence[str], config: TreeFeatureConfig | None = None) -> None:
        self.feature_names = list(feature_names)
        self.config = config or TreeFeatureConfig()
        self.n_features_in_ = len(self.feature_names)
        self.column_indices_: np.ndarray = np.zeros(0, dtype=int)
        self.cuts_: list[np.ndarray] = []
        self.schema_: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.n_bins_: np.ndarray = np.zeros(0, dtype=int)

    # ------------------------------------------------------------------ fit
    def fit(self, x: Any) -> "TreeBinner":
        """Learn column selection and cut points from training rows."""
        x = _validate_matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("x has a different number of columns than feature_names")
        cfg = self.config
        dropped = set(cfg.drop_columns)
        schema_all = schema_for_names(self.feature_names)
        rng = np.random.default_rng(cfg.seed)
        kept: list[int] = []
        cuts: list[np.ndarray] = []
        for j, name in enumerate(self.feature_names):
            if name in dropped:
                continue
            col = _clean_column(x[:, j], *schema_all[j])
            observed = ~np.isnan(col)
            values = col[observed]
            distinct = np.unique(values)
            if len(distinct) == 0 or (len(distinct) == 1 and observed.all()):
                continue  # no information: all missing, or constant without missingness
            if len(distinct) <= cfg.max_bins:
                edges = (distinct[:-1] + distinct[1:]) / 2.0
            else:
                sample = (
                    values if len(values) <= cfg.sample_size else rng.choice(values, cfg.sample_size, replace=False)
                )
                edges = np.unique(np.quantile(sample, np.arange(1, cfg.max_bins) / cfg.max_bins))
            kept.append(j)
            cuts.append(np.asarray(edges, dtype=float))
        self.column_indices_ = np.asarray(kept, dtype=int)
        self.cuts_ = cuts
        self.schema_ = [tuple(map(tuple, schema_all[j])) for j in kept]
        self.n_bins_ = np.asarray([len(c) + 2 for c in cuts], dtype=int)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, x: Any) -> np.ndarray:
        """Bin rows with the fitted cuts; returns ``uint8`` of shape (n, n_out)."""
        if not len(self.column_indices_):
            raise ValueError("TreeBinner must be fitted before transform")
        x = _validate_matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("x has a different number of columns than the fitted data")
        out = np.empty((x.shape[0], len(self.column_indices_)), dtype=np.uint8)
        for start in range(0, x.shape[0], self.config.chunk_rows):
            stop = min(start + self.config.chunk_rows, x.shape[0])
            block = x[start:stop]
            for k, j in enumerate(self.column_indices_):
                col = _clean_column(block[:, j], *self.schema_[k])
                missing = np.isnan(col)
                bins = np.searchsorted(self.cuts_[k], col, side="right") + 1
                bins[missing] = MISSING_BIN
                out[start:stop, k] = bins
        return out

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable fitted state."""
        return {
            "feature_names": list(self.feature_names),
            "config": asdict(self.config),
            "column_indices": self.column_indices_.tolist(),
            "cuts": [c.tolist() for c in self.cuts_],
            "schema": [[list(m), list(z)] for m, z in self.schema_],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "TreeBinner":
        """Rebuild a fitted binner from :meth:`to_dict` output."""
        binner = cls(state["feature_names"], tree_feature_config(state["config"]))
        binner.column_indices_ = np.asarray(state["column_indices"], dtype=int)
        binner.cuts_ = [np.asarray(c, dtype=float) for c in state["cuts"]]
        binner.schema_ = [(tuple(m), tuple(z)) for m, z in state["schema"]]
        binner.n_bins_ = np.asarray([len(c) + 2 for c in binner.cuts_], dtype=int)
        return binner


@dataclass(frozen=True)
class WideFeatureConfig:
    """Options of the wide preset."""

    drop_columns: tuple[str, ...] = IDENTIFIER_COLUMNS
    max_card: int = 30
    nominal: tuple[str, ...] = ("_STATE", "EXRACT11", "EXRACT21")
    min_level_share: float = 0.001
    n_quantile_bins: int = 10
    clip: tuple[float, float] = (0.005, 0.995)
    chunk_rows: int = 16_384  # rows per block; also fixes the Newton solver's summation order

    def __post_init__(self) -> None:
        if self.max_card < 2 or self.n_quantile_bins < 1 or self.chunk_rows < 1:
            raise ValueError("max_card >= 2, n_quantile_bins >= 1 and chunk_rows >= 1 required")
        if not 0 <= self.min_level_share < 1 or not 0 <= self.clip[0] < self.clip[1] <= 1:
            raise ValueError("min_level_share in [0, 1) and 0 <= clip[0] < clip[1] <= 1 required")


class WideFeatureMap:
    """Fit on training rows, then ``transform_chunks`` any rows."""

    def __init__(self, feature_names: Sequence[str], config: WideFeatureConfig | None = None) -> None:
        self.feature_names = list(feature_names)
        self.config = config or WideFeatureConfig()
        self.columns_: list[dict[str, Any]] = []
        self.feature_names_out_: list[str] = ["bias"]
        self.n_features_out_ = 1

    def fit(self, x: Any) -> "WideFeatureMap":
        """Learn levels, clipping bounds, standardisation and quantile edges."""
        x = _validate_matrix(x)
        if x.shape[1] != len(self.feature_names):
            raise ValueError("x has a different number of columns than feature_names")
        cfg = self.config
        dropped = set(cfg.drop_columns)
        schema = schema_for_names(self.feature_names)
        n_rows = x.shape[0]
        self.columns_, names = [], ["bias"]
        for j, name in enumerate(self.feature_names):
            if name in dropped:
                continue
            col = _clean_column(x[:, j], *schema[j])
            observed = ~np.isnan(col)
            values = col[observed]
            distinct, counts = np.unique(values, return_counts=True)
            if len(distinct) == 0 or (len(distinct) == 1 and observed.all()):
                continue
            spec: dict[str, Any] = {
                "index": j,
                "name": name,
                "has_missing": bool((~observed).any()),
                "schema": [list(schema[j][0]), list(schema[j][1])],
            }
            if name in cfg.nominal or len(distinct) <= cfg.max_card:
                keep = counts / n_rows >= cfg.min_level_share
                # ``other`` always exists so rare and unseen levels have a column.
                spec.update(kind="categorical", levels=distinct[keep].tolist())
                names += [f"{name}={level:g}" for level in spec["levels"]] + [f"{name}=other"]
            else:
                lo, hi = (float(v) for v in np.quantile(values, cfg.clip))
                clipped = np.clip(values, lo, hi)
                std = float(clipped.std()) or 1.0
                edges = np.unique(np.quantile(values, np.arange(1, cfg.n_quantile_bins) / cfg.n_quantile_bins))
                spec.update(kind="numeric", clip=[lo, hi], mean=float(clipped.mean()), std=std, edges=edges.tolist())
                names += [name] + [f"{name}@q{i}" for i in range(len(edges) + 1)]
            names += [f"{name}=missing"] if spec["has_missing"] else []
            self.columns_.append(spec)
        self.feature_names_out_ = names
        self.n_features_out_ = len(names)
        return self

    def _fill(self, block: np.ndarray) -> np.ndarray:
        out = np.zeros((block.shape[0], self.n_features_out_), dtype=np.float32)
        out[:, 0] = 1.0
        position = 1
        for spec in self.columns_:
            col = _clean_column(block[:, spec["index"]], *spec["schema"])
            missing = np.isnan(col)
            if spec["kind"] == "categorical":
                levels = np.asarray(spec["levels"], dtype=float)
                for level in levels:
                    out[:, position] = col == level
                    position += 1
                out[:, position] = ~missing & ~np.isin(col, levels)
                position += 1
            else:
                z = (np.clip(col, *spec["clip"]) - spec["mean"]) / spec["std"]
                out[:, position] = np.where(missing, 0.0, z)
                position += 1
                edges = np.asarray(spec["edges"], dtype=float)
                rows = np.flatnonzero(~missing)
                out[rows, position + np.searchsorted(edges, col[rows], side="right")] = 1.0
                position += len(edges) + 1
            if spec["has_missing"]:
                out[:, position] = missing
                position += 1
        return out

    def transform_chunks(self, x: Any) -> Iterator[np.ndarray]:
        """Yield ``float32`` blocks of at most ``chunk_rows`` rows."""
        x = _validate_matrix(x)
        if x.shape[1] != len(self.feature_names):
            raise ValueError("x has a different number of columns than the fitted data")
        for start in range(0, x.shape[0], self.config.chunk_rows):
            yield self._fill(x[start : start + self.config.chunk_rows])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable fitted state."""
        return {
            "feature_names": list(self.feature_names),
            "config": asdict(self.config),
            "columns": self.columns_,
            "feature_names_out": self.feature_names_out_,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "WideFeatureMap":
        """Rebuild a fitted map from :meth:`to_dict` output."""
        fmap = cls(state["feature_names"], wide_feature_config(state["config"]))
        fmap.columns_ = [dict(spec) for spec in state["columns"]]
        fmap.feature_names_out_ = list(state["feature_names_out"])
        fmap.n_features_out_ = len(fmap.feature_names_out_)
        return fmap


def tree_feature_config(spec: Mapping[str, Any]) -> TreeFeatureConfig:
    """``TreeFeatureConfig`` from a JSON dict (lists become tuples)."""
    spec = dict(spec)
    if "drop_columns" in spec:
        spec["drop_columns"] = tuple(spec["drop_columns"])
    return TreeFeatureConfig(**spec)


def wide_feature_config(spec: Mapping[str, Any]) -> WideFeatureConfig:
    """``WideFeatureConfig`` from a JSON dict (lists become tuples)."""
    spec = dict(spec)
    for key in ("drop_columns", "nominal", "clip"):
        if key in spec:
            spec[key] = tuple(spec[key])
    return WideFeatureConfig(**spec)
