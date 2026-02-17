# BAB III: Metodologi Penelitian (Ringkas dan Operasional)

Tujuan: Mendeteksi kesesuaian dan penyimpangan jalur karier mahasiswa dengan pendekatan semantik murni menggunakan embedding BERT (tanpa TF-IDF/korpus klasik), serta memberikan rekomendasi alternatif.

3.1 Arsitektur Sistem
- Input: Teks CV mahasiswa dan profil karier pasar kerja.
- Preprocessing: Normalisasi ringan (whitespace, trimming, optional lowercase) untuk menjaga makna.
- Encoding: Pretrained sentence-transformer (multilingual). Output vektor ter-normalisasi L2.
- Similarity: Cosine similarity antara CV dan profil karier.
- Ranking: Urutan karier berdasarkan similarity menurun.
- Drift Analysis: Bandingkan kesesuaian declared interest vs alternatif terbaik; kategorikan Aligned/Minor/Major.
- Recommendation: Top-k alternatif di atas ambang minimal similarity.

3.2 Model dan Alasan Pemilihan
- Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
- Alasan: Multilingual (mendukung Bahasa Indonesia), ringan untuk CPU, embedding kalimat berkualitas, reproducible tanpa fine-tuning.

3.3 Definisi Operasional
- Representasi Semantik: vektor d-dimensi hasil encoder; relasi makna dibandingkan melalui sudut antar vektor.
- Cosine Similarity: s(x,y) = (x·y) / (||x|| ||y||), nilai [0,1] karena normalisasi L2.
- Drift: advantage = s(best_alt) − s(declared). Kategori:
  - Aligned: s(declared) ≥ τ_high dan advantage ≤ δ_minor
  - Minor Drift: s(declared) ≥ τ_mid dan advantage > δ_minor
  - Major Drift: s(declared) < τ_mid dan s(best_alt) ≥ τ_high
  - Moderate Fit: kasus lain (borderline)
- Rekomendasi: daftar karier dengan s ≥ θ_min dan top-k.

3.4 Parameter dan Threshold
- τ_high = 0.70, τ_mid = 0.60, δ_minor = 0.08, θ_min = 0.55.
- Penentuan awal secara heuristik berdasarkan literatur umum dan uji coba kecil; dapat di-tuning menggunakan validasi silang sederhana pada data internal.

3.5 Validasi dan Evaluasi
- Deskriptif: Distribusi similarity per mahasiswa; proporsi kategori drift.
- Case Study: Uraikan beberapa contoh nyata, interpretasi rasional.
- Robustness: Uji sensitivitas threshold; bandingkan variasi τ/δ/θ.
- Reproducibility: Catat versi model, seed, konfigurasi pipeline.

3.6 Batasan
- Tidak ada fine-tuning: akurasi domain-spesifik terbatas.
- Bergantung pada kualitas teks CV dan deskripsi karier.
- Threshold bersifat heuristik; bukan klasifikasi supervised.

# BAB IV: Implementasi dan Hasil

4.1 Lingkungan Implementasi
- Bahasa: Python 3.10+
- Dependensi: sentence-transformers, numpy.
- Perangkat: CPU laptop standar.

4.2 Implementasi
- Struktur modul:
  - src/preprocess.py: fungsi preprocess_text, preprocess_batch
  - src/embedding.py: load_model, embed_texts
  - src/similarity.py: cosine_similarity_matrix, rank_topk
  - src/drift.py: analyze_drift (kategorisasi drift)
  - src/recommender.py: recommend_alternatives
  - app.py: CLI pipeline end-to-end

4.3 Dataset
- data/careers.json: daftar profil karier (id, title, description, skills)
- data/students.json: data mahasiswa (id, name, cv_text, declared_interest)
- Format JSON sederhana agar mudah direplikasi.

4.4 Prosedur Eksekusi
- Instal dependencies: pip install -r requirements.txt
- Jalankan: python app.py --cv data/students.json --careers data/careers.json --topk 5 --min-sim 0.55
- Opsi model: --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

4.5 Contoh Hasil dan Interpretasi
- Output per mahasiswa:
  - Ranking karier dengan skor similarity.
  - Drift Analysis: status, s(declared), s(best_alt), advantage, rationale.
  - Recommended Alternatives: daftar karier dengan s ≥ θ_min.
- Contoh interpretasi:
  - Jika Andi memiliki s(declared)=0.68 dan best_alt=0.74, advantage=0.06 > δ_minor? Tidak (0.06 ≤ 0.08) → Aligned borderline. Rationale: Deklarasi cukup kuat, alternatif tidak jauh lebih baik.
  - Jika Dewi s(declared)=0.58 dan best_alt=0.73 → Major Drift (declared rendah, alternatif tinggi). Rekomendasi fokus pada alternatif dengan skor tinggi.

4.6 Kompleksitas dan Efisiensi
- Encoding kompleksitas O(N+M) untuk N teks CV, M profil karier.
- Similarity komputasi matriks O(NM d); praktis karena d ≈ 384 untuk MiniLM dan N,M kecil-menengah.
- Memori: matriks similarity N×M float32.

4.7 Analisis Sensitivitas Parameter
- Variasi τ_high: 0.68–0.75; efek pada proporsi Aligned.
- Variasi δ_minor: 0.05–0.12; efek pada Minor vs Aligned.
- Variasi θ_min: 0.50–0.60; efek jumlah rekomendasi.
- Dokumentasikan perubahan kategori untuk transparansi.

4.8 Reproducibility Checklist
- Catat versi model dan hash.
- Simpan konfigurasi CLI yang digunakan.
- Seed tidak relevan (inference deterministik), tetapi catat versi library.

4.9 Keterlacakan dan Interpretabilitas
- Simpan log skor untuk audit.
- Sertakan rationale per kategori drift (otomatis dari fungsi analyze_drift).

4.10 Keterbatasan Hasil
- Perbedaan bahasa campuran dapat mempengaruhi embedding.
- Deskripsi karier terlalu singkat dapat menurunkan akurasi semantik.
