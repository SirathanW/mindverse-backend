import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());            // อนุญาตทุก origin (พอสำหรับ dev บน web)
app.use(express.json());

const HF_TOKEN = process.env.HF_TOKEN;
if (!HF_TOKEN) console.warn("⚠️  Missing HF_TOKEN in .env");

const HF_API =process.env.HF_API;
//const HF_API =
//  "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill";

app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body?.input ?? "";
    const r = await fetch(HF_API, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: userMessage }),
    });
    const data = await r.json();
    const reply =
      data.generated_text ||
      data[0]?.generated_text ||
      data?.conversation?.generated_responses?.at(-1) ||
      "No response from AI";
    res.json({ reply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "server_error", message: String(err) });
  }
});

app.post("/horoscope", async (req, res) => {
  try {
    const { name, birthDate, birthTime, language } = req.body ?? {};
    const prompt = `
Create a single consolidated horoscope (love, career, health, wealth).
Name: ${name ?? "User"}
Birth Date: ${birthDate ?? "-"}
Birth Time: ${birthTime ?? "-"}
Language: ${language ?? "English"}
`;
    const r = await fetch(HF_API, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: prompt }),
    });
    const data = await r.json();
    const text = data.generated_text || data[0]?.generated_text || "No prediction available";
    res.json({ text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "server_error", message: String(err) });
  }
});

const PORT = process.env.PORT || 8787;
app.listen(PORT, () => {
  console.log(`✅ Proxy running on http://localhost:${PORT}`);
});