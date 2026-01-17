use anyhow::Result;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "content")]
pub(crate) enum Frame {
    Handshake { payload: HandshakePayload },
    HandshakeAck { payload: HandshakeAckPayload },
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub(crate) struct HandshakePayload {
    pub(crate) sfn_name: String,
    pub(crate) credential: Option<String>,
    pub(crate) metadata: Vec<u8>,
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub(crate) struct HandshakeAckPayload {
    pub(crate) ok: bool,
    pub(crate) reason: Option<String>,
}

pub(crate) async fn read_frame(reader: &mut (impl AsyncReadExt + Unpin)) -> Result<Frame> {
    let length = reader.read_u32().await?;
    let mut data = vec![0; length as usize];
    reader.read_exact(&mut data).await?;
    let f: Frame = serde_json::from_slice(&data)?;
    debug!("read frame: {:?}", f);
    trace!("read frame raw: {}", String::from_utf8_lossy(&data));
    Ok(f)
}

pub(crate) async fn write_frame(
    writer: &mut (impl AsyncWriteExt + Unpin),
    frame: &Frame,
) -> Result<()> {
    let data = serde_json::to_vec(frame)?;
    writer.write_u32(data.len() as u32).await?;
    writer.write_all(&data).await?;
    debug!("write frame: {:?}", frame);
    trace!("write frame raw: {}", String::from_utf8_lossy(&data));
    Ok(())
}
