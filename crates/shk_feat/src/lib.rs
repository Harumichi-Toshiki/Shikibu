//! Shikibu の特徴抽出（M1）。Python からは `shikibu_feat` として呼ぶ。
//!
//! 仕様は `docs/DESIGN.md` の第F部（凍結区画）。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod bench;
mod feature;
mod halfk;
mod kingzone;
mod position;
mod psv;
mod raypair;
mod loader;
mod rng;
mod shuffle;
mod stats;

fn to_py_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// 特徴の版（一覧は shikibu.networks.FEATURES と docs/FORMATS.md）。
fn check_version(version: i64) -> PyResult<u8> {
    match version {
        0..=13 => Ok(version as u8),
        _ => Err(PyValueError::new_err(format!(
            "version は 0〜13（0・1 = RayPair、2〜11 = HalfK 系、12・13 = RayPair ＋ 玉の周り）: {version}"
        ))),
    }
}

fn check_perspective(perspective: i64) -> PyResult<u8> {
    match perspective {
        0 | 1 => Ok(perspective as u8),
        _ => Err(PyValueError::new_err(format!(
            "perspective は 0 か 1: {perspective}"
        ))),
    }
}

/// 特徴量の次元数（version は check_version と同じ）。
#[pyfunction]
fn num_features(version: i64) -> PyResult<u32> {
    feature::num_features(check_version(version)?).map_err(to_py_err)
}

/// SFEN から特徴の多重集合（昇順・重複込み）を返す。
#[pyfunction]
fn extract_sfen(sfen: &str, perspective: i64, version: i64) -> PyResult<Vec<u32>> {
    let persp = check_perspective(perspective)?;
    let version = check_version(version)?;
    let pos = position::parse_sfen(sfen).map_err(to_py_err)?;
    feature::features(&pos, persp, version).map_err(to_py_err)
}

/// 玉のマス（RayPair の視点座標 `y × 9 + x`）。戻り値は (手番側の玉, 相手の玉)。
#[pyfunction]
fn king_squares_sfen(sfen: &str) -> PyResult<(u32, u32)> {
    let pos = position::parse_sfen(sfen).map_err(to_py_err)?;
    feature::king_squares(&pos).map_err(to_py_err)
}

/// PSV（40バイト）を読む。戻り値は (sfen, score, move, gamePly, game_result)。
#[pyfunction]
fn decode_psv(rec: &[u8]) -> PyResult<(String, i16, u16, u16, i8)> {
    let (pos, score, mv, ply, result) = position::decode_psv(rec).map_err(to_py_err)?;
    Ok((position::to_sfen(&pos, ply), score, mv, ply, result))
}

/// PSV から特徴の多重集合（昇順・重複込み）を返す。
#[pyfunction]
fn extract_psv(rec: &[u8], perspective: i64, version: i64) -> PyResult<Vec<u32>> {
    let persp = check_perspective(perspective)?;
    let version = check_version(version)?;
    let (pos, _, _, _, _) = position::decode_psv(rec).map_err(to_py_err)?;
    feature::features(&pos, persp, version).map_err(to_py_err)
}

/// 計測用：PSV の先頭 `n` 局面（足りなければ循環）について、解読と両視点の特徴抽出を行い、
/// `{"positions": n, "seconds": 秒}` を返す。Python を介さない純粋な計算の速さを測る。
#[pyfunction]
fn bench_extract<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    n: usize,
    version: i64,
    num_threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let version = check_version(version)?;
    if n == 0 {
        return Err(PyValueError::new_err("n は1以上"));
    }
    if num_threads == 0 {
        return Err(PyValueError::new_err("num_threads は1以上"));
    }
    let seconds = py
        .allow_threads(|| bench::bench_extract(&paths, n, version, num_threads))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("positions", n)?;
    d.set_item("seconds", seconds)?;
    Ok(d)
}

/// 全局面をシャッフルしてシャードに分割する（§S3.1）。出力はシャードのパス（昇順）。
#[pyfunction]
fn shuffle_shards(
    py: Python<'_>,
    inputs: Vec<String>,
    out_dir: String,
    records_per_shard: u64,
    seed: u64,
    mem_limit_mb: u64,
) -> PyResult<Vec<String>> {
    py.allow_threads(|| {
        shuffle::shuffle_shards(&inputs, &out_dir, records_per_shard, seed, mem_limit_mb)
    })
    .map_err(to_py_err)
}

#[pymodule]
fn shikibu_feat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(num_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sfen, m)?)?;
    m.add_function(wrap_pyfunction!(decode_psv, m)?)?;
    m.add_function(wrap_pyfunction!(king_squares_sfen, m)?)?;
    m.add_function(wrap_pyfunction!(extract_psv, m)?)?;
    m.add_function(wrap_pyfunction!(bench_extract, m)?)?;
    m.add_function(wrap_pyfunction!(stats::psv_stats, m)?)?;
    m.add_function(wrap_pyfunction!(stats::estimate_eval_scale, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_shards, m)?)?;
    m.add_class::<loader::BatchLoader>()?;
    Ok(())
}
