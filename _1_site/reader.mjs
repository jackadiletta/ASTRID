import { trajectory } from './trajectory.mjs';
self.onmessage = async ({ data: url }) => {
  try {
    const { parquetRead, parquetMetadata, parquetSchema } = await import('https://cdn.jsdelivr.net/npm/hyparquet@1.32.0/src/index.js');
    const { compressors } = await import('https://cdn.jsdelivr.net/npm/hyparquet-compressors@1.1.1/+esm');
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Parquet request failed (${response.status}). Check the manifest path.`);
    const file = await response.arrayBuffer();
    const metadata = parquetMetadata(file);
    const available = parquetSchema(metadata).children.map(c => c.element.name);
    const required = ['lat', 'lon', 'accl_x', 'accl_y', 'accl_z'];
    for (const name of required) if (!available.includes(name)) throw new Error(`Missing Parquet column: ${name}`);
    let indexName;
    const pandas = metadata.key_value_metadata?.find(v => v.key === 'pandas');
    if (pandas?.value) {
      const info = JSON.parse(pandas.value);
      indexName = info.index_columns?.find(v => typeof v === 'string');
    }
    const columns = [...required, ...(['time', indexName].filter(v => v && available.includes(v)))];
    const values = {};
    await parquetRead({ file, metadata, columns, compressors, onChunk({ columnName, columnData, rowStart }) {
      const target = values[columnName] ||= [];
      for (let i = 0; i < columnData.length; i++) target[rowStart + i] = columnData[i];
    }});
    if (indexName) values.index = values[indexName].map(Number);
    self.postMessage({ result: trajectory(values) });
  } catch (error) { self.postMessage({ error: error.message }); }
};
