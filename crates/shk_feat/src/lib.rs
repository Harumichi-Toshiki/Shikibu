//! Shikibu の特徴抽出（M1）。Python からは `shikibu_feat` として呼ぶ。
//!
//! 仕様は `docs/DESIGN.md` の第F部（凍結区画）。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod position;
mod psv;
mod raypair;
mod rng;
mod stats;

fn to_py_err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

fn check_version(version: i64) -> PyResult<u8> {
    match version {
        0 | 1 => Ok(version as u8),
        _ => Err(PyValueError::new_err(format!(
            "version は 0 か 1: {version}"
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

/// 特徴量の次元数。version: 0 = RayPair_v0、1 = RayPair_v1。
#[pyfunction]
fn num_features(version: i64) -> PyResult<u32> {
    raypair::num_features(check_version(version)?).map_err(to_py_err)
}

/// SFEN から特徴の多重集合（昇順・重複込み）を返す。
#[pyfunction]
fn extract_sfen(sfen: &str, perspective: i64, version: i64) -> PyResult<Vec<u32>> {
    let persp = check_perspective(perspective)?;
    let version = check_version(version)?;
    let pos = position::parse_sfen(sfen).map_err(to_py_err)?;
    raypair::features(&pos, persp, version).map_err(to_py_err)
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
    raypair::features(&pos, persp, version).map_err(to_py_err)
}

#[pymodule]
fn shikibu_feat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(num_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sfen, m)?)?;
    m.add_function(wrap_pyfunction!(decode_psv, m)?)?;
    m.add_function(wrap_pyfunction!(extract_psv, m)?)?;
    m.add_function(wrap_pyfunction!(stats::psv_stats, m)?)?;
    m.add_function(wrap_pyfunction!(stats::estimate_eval_scale, m)?)?;
    Ok(())
}
