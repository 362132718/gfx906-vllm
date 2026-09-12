const KB = 1024;
const kvPerTokCard = (4 * 256 * 2 * 2 * 16) / 2;
console.log("KV bytes/tok/card", kvPerTokCard, "=", kvPerTokCard / KB, "KiB");
const BW = 1024e9;

const marg = 16205 - 2056;
const dt = 35.02 - 29.98;
const floorMarg = (marg * kvPerTokCard) / BW * 1000;
console.log("marginal tokens", marg, "dt", dt.toFixed(3), "ms floor", floorMarg.toFixed(4), "ms util%", ((floorMarg / dt) * 100).toFixed(2));

const short = 2050;
const floorShort = (short * kvPerTokCard) / BW * 1e6;
console.log("short ctx floor us", floorShort.toFixed(1), "util vs 513.6 =", ((floorShort / 513.6) * 100).toFixed(2), "%");

const attn16 = 513.6 + dt * 1000;
const floor16 = (16205 * kvPerTokCard) / BW * 1e6;
const stepCeil = 35.02 - attn16 / 1000 + floor16 / 1000;
console.log("attn@16K us", attn16.toFixed(0), "share%", ((attn16 / 35020) * 100).toFixed(2));
console.log("floor@16K us", floor16.toFixed(0), "ceiling tok/s", (1000 / stepCeil).toFixed(2), "gain%", ((1000 / stepCeil / 28.56 - 1) * 100).toFixed(2));

console.log("slopes ms/1k: 2->8K", ((31.30 - 29.98) / ((8208 - 2056) / 1000)).toFixed(3), " 8->16K", ((35.02 - 31.30) / ((16205 - 8208) / 1000)).toFixed(3));
console.log("MiB marginal", ((marg * kvPerTokCard) / (1024 * 1024)).toFixed(1));
