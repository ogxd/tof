//! Binary blob format for serializing compressed latent representations.
//!
//! Format (all multi-byte integers are big-endian):
//! ```text
//! [magic:   4 bytes]  0x4E 0x43 0x4C 0x00  ("NCL\0")
//! [version: 1 byte ]  0x01
//! [shape_h: 4 bytes]  latent height
//! [shape_w: 4 bytes]  latent width
//! [n_streams: 4 bytes]
//! for each stream:
//!     [len: 4 bytes]
//!     [data: `len` bytes]
//! ```

use anyhow::{bail, ensure, Context, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

const MAGIC: [u8; 4] = [0x4E, 0x43, 0x4C, 0x00]; // "NCL\0"
const VERSION: u8 = 1;

/// The compressed representation of a single image.
#[derive(Debug, Clone)]
pub struct CompressedBlob {
    /// Spatial dimensions of the latent grid (height, width).
    pub latent_shape: (u32, u32),
    /// Entropy-coded byte streams (typically: y_strings, z_strings).
    pub streams: Vec<Vec<u8>>,
}

impl CompressedBlob {
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.write_all(&MAGIC)?;
        buf.write_u8(VERSION)?;
        buf.write_u32::<BigEndian>(self.latent_shape.0)?;
        buf.write_u32::<BigEndian>(self.latent_shape.1)?;
        buf.write_u32::<BigEndian>(self.streams.len() as u32)?;
        for stream in &self.streams {
            buf.write_u32::<BigEndian>(stream.len() as u32)?;
            buf.write_all(stream)?;
        }
        Ok(buf)
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);

        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)
            .context("failed to read blob magic")?;
        ensure!(magic == MAGIC, "invalid blob magic: expected NCL\\0");

        let version = cur.read_u8().context("failed to read blob version")?;
        ensure!(
            version == VERSION,
            "unsupported blob version {version}, expected {VERSION}"
        );

        let shape_h = cur
            .read_u32::<BigEndian>()
            .context("failed to read latent height")?;
        let shape_w = cur
            .read_u32::<BigEndian>()
            .context("failed to read latent width")?;

        let n_streams = cur
            .read_u32::<BigEndian>()
            .context("failed to read stream count")?;
        if n_streams > 64 {
            bail!("unreasonable stream count: {n_streams}");
        }

        let mut streams = Vec::with_capacity(n_streams as usize);
        for i in 0..n_streams {
            let len = cur
                .read_u32::<BigEndian>()
                .with_context(|| format!("failed to read length for stream {i}"))?;
            let mut data = vec![0u8; len as usize];
            cur.read_exact(&mut data)
                .with_context(|| format!("failed to read data for stream {i}"))?;
            streams.push(data);
        }

        Ok(Self {
            latent_shape: (shape_h, shape_w),
            streams,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let blob = CompressedBlob {
            latent_shape: (16, 24),
            streams: vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8]],
        };
        let bytes = blob.to_bytes().unwrap();
        let decoded = CompressedBlob::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.latent_shape, (16, 24));
        assert_eq!(decoded.streams.len(), 2);
        assert_eq!(decoded.streams[0], vec![1, 2, 3]);
        assert_eq!(decoded.streams[1], vec![4, 5, 6, 7, 8]);
    }
}
