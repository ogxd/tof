use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;

/// A record representing a single compressed photo in the catalog.
#[derive(Debug, Clone)]
pub struct PhotoRecord {
    pub id: i64,
    pub original_path: String,
    pub filename: String,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
    pub original_size_bytes: u64,
    pub format: String,
    pub latent_path: String,
    pub compressed_size_bytes: u64,
    pub compression_ratio: f64,
    pub psnr: Option<f64>,
    pub ms_ssim: Option<f64>,
    pub exif_json: Option<String>,
    pub ingested_at: String,
    pub model_name: String,
    pub quality_level: u8,
    pub pad_h: u32,
    pub pad_w: u32,
}

/// Fields needed to insert a new photo record (id and ingested_at are auto-generated).
#[derive(Debug)]
pub struct NewPhotoRecord {
    pub original_path: String,
    pub filename: String,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
    pub original_size_bytes: u64,
    pub format: String,
    pub latent_path: String,
    pub compressed_size_bytes: u64,
    pub compression_ratio: f64,
    pub psnr: Option<f64>,
    pub ms_ssim: Option<f64>,
    pub exif_json: Option<String>,
    pub model_name: String,
    pub quality_level: u8,
    pub pad_h: u32,
    pub pad_w: u32,
}

/// Thin wrapper around a SQLite connection for the photo catalog.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) the catalog database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS photos (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path       TEXT    NOT NULL,
                filename            TEXT    NOT NULL,
                sha256              TEXT    UNIQUE NOT NULL,
                width               INTEGER NOT NULL,
                height              INTEGER NOT NULL,
                original_size_bytes INTEGER NOT NULL,
                format              TEXT    NOT NULL,
                latent_path         TEXT    NOT NULL,
                compressed_size_bytes INTEGER NOT NULL,
                compression_ratio   REAL    NOT NULL,
                psnr                REAL,
                ms_ssim             REAL,
                exif_json           TEXT,
                ingested_at         TEXT    NOT NULL DEFAULT (datetime('now')),
                model_name          TEXT    NOT NULL,
                quality_level       INTEGER NOT NULL,
                pad_h               INTEGER NOT NULL DEFAULT 0,
                pad_w               INTEGER NOT NULL DEFAULT 0
            );",
        )?;

        Ok(Self { conn })
    }

    /// Check whether a photo with the given SHA-256 already exists.
    pub fn exists_by_sha256(&self, sha256: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM photos WHERE sha256 = ?1",
            params![sha256],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Insert a new photo record, returning its assigned ID.
    pub fn insert_photo(&self, rec: &NewPhotoRecord) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO photos (
                original_path, filename, sha256,
                width, height, original_size_bytes, format,
                latent_path, compressed_size_bytes, compression_ratio,
                psnr, ms_ssim, exif_json,
                model_name, quality_level, pad_h, pad_w
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
            params![
                rec.original_path,
                rec.filename,
                rec.sha256,
                rec.width,
                rec.height,
                rec.original_size_bytes,
                rec.format,
                rec.latent_path,
                rec.compressed_size_bytes,
                rec.compression_ratio,
                rec.psnr,
                rec.ms_ssim,
                rec.exif_json,
                rec.model_name,
                rec.quality_level,
                rec.pad_h,
                rec.pad_w,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Retrieve a photo record by ID.
    pub fn get_photo(&self, id: i64) -> Result<Option<PhotoRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, original_path, filename, sha256,
                    width, height, original_size_bytes, format,
                    latent_path, compressed_size_bytes, compression_ratio,
                    psnr, ms_ssim, exif_json, ingested_at,
                    model_name, quality_level, pad_h, pad_w
             FROM photos WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], row_to_record)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// List photos with an optional limit.
    pub fn list_photos(&self, limit: u32) -> Result<Vec<PhotoRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, original_path, filename, sha256,
                    width, height, original_size_bytes, format,
                    latent_path, compressed_size_bytes, compression_ratio,
                    psnr, ms_ssim, exif_json, ingested_at,
                    model_name, quality_level, pad_h, pad_w
             FROM photos ORDER BY id LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], row_to_record)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Get all photo records (for stats computation).
    pub fn get_all_photos(&self) -> Result<Vec<PhotoRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, original_path, filename, sha256,
                    width, height, original_size_bytes, format,
                    latent_path, compressed_size_bytes, compression_ratio,
                    psnr, ms_ssim, exif_json, ingested_at,
                    model_name, quality_level, pad_h, pad_w
             FROM photos ORDER BY id",
        )?;
        let rows = stmt.query_map([], row_to_record)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Get the count of stored photos.
    pub fn count(&self) -> Result<u64> {
        let count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM photos", [], |row| row.get(0))?;
        Ok(count as u64)
    }

    /// Begin a transaction — returns a handle that must be committed.
    pub fn transaction(&mut self) -> Result<rusqlite::Transaction<'_>> {
        Ok(self.conn.transaction()?)
    }
}

fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<PhotoRecord> {
    Ok(PhotoRecord {
        id: row.get(0)?,
        original_path: row.get(1)?,
        filename: row.get(2)?,
        sha256: row.get(3)?,
        width: row.get(4)?,
        height: row.get(5)?,
        original_size_bytes: row.get(6)?,
        format: row.get(7)?,
        latent_path: row.get(8)?,
        compressed_size_bytes: row.get(9)?,
        compression_ratio: row.get(10)?,
        psnr: row.get(11)?,
        ms_ssim: row.get(12)?,
        exif_json: row.get(13)?,
        ingested_at: row.get(14)?,
        model_name: row.get(15)?,
        quality_level: row.get(16)?,
        pad_h: row.get(17)?,
        pad_w: row.get(18)?,
    })
}
