
import google.generativeai as genai
import streamlit as st
genai.configure(api_key=st.secrets['GEMINI_TOKEN'])

prompt = """
      Prompt:
      IndiHome adalah layanan internet yang populer di Indonesia. Beberapa pertanyaan umum terkait IndiHome sering muncul dari pelanggan yang ingin mengetahui lebih lanjut tentang layanan tersebut. Di bawah ini adalah daftar pertanyaan yang sering diajukan:

      1. Apakah IndiHome pindah ke Telkomsel?
      2. Apakah kelebihan IndiHome setelah dialihkan ke Telkomsel?
      3. Apakah pelanggan lama IndiHome harus daftar ulang ke Telkomsel?
      4. Apakah ada perubahan cara bayar (nomor rekening/bank)?
      5. Apakah ada perubahan produk/tarif layanan IndiHome?
      6. Apakah billing/tagihan IndiHome akan disatukan dengan tagihan Telkomsel?
      7. Saya pelanggan Telkomsel dan IndiHome, apakah terdapat penawaran bundling?
      8. Apakah aplikasi myIndiHome akan digabung dengan MyTelkomsel?
      9. Apakah telepon rumah saya akan dipindahkan ke Telkomsel?
      10. Apakah GraPARI Telkomsel melayani IndiHome?

      Berikut adalah jawaban potensial untuk setiap pertanyaan:
      1. Ya, mulai 1 Juli 2023 layanan IndiHome secara resmi bergabung menjadi bagian dari Telkomsel.
      2. Dengan mengintegrasikan layanan IndiHome yang menawarkan koneksi internet tetap melalui serat optik serta dengan jaringan seluler broadband yang luas, memungkinkan pelanggan IndiHome dan Telkomsel untuk mendapatkan pengalaman konektivitas broadband yang mulus dan pengalaman digital yang lebih baik di dalam maupun di luar rumah, tanpa terikat pada satu teknologi jaringan tertentu. Ke depannya, Telkomsel juga akan mengembangkan beragam penawaran produk dan layanan yang lebih terjangkau dan bernilai tambah, yang mengintegrasikan seluruh keunggulan, baik layanan fixed broadband IndiHome maupun layanan mobile broadband Telkomsel, seperti Orbit, Telkomsel PraBayar, dan Halo.
      3. Tidak, pelanggan IndiHome yang dialihkan ke Telkomsel tidak perlu melakukan daftar ulang, layanan IndiHome yang digunakan akan terdaftar dalam pengelolaan Telkomsel.
      4. Tidak ada perubahan cara pembayaran billing/tagihan IndiHome setelah IndiHome bergabung dengan Telkomsel.
      5. Tidak ada perubahan paket produk/tarif layanan IndiHome setelah IndiHome bergabung dengan Telkomsel. Pelanggan akan mendapatkan informasi secara berkala jika terdapat perubahan atau penawaran produk lebih lanjut.
      6. Billing/tagihan IndiHome saat ini belum disatukan dengan Telkomsel Halo. Kami akan menginformasikan apabila ada perubahan tagihan IndiHome dan Telkomsel.
      7. Saat ini telah hadir paket Telkomsel One, sebagai penawaran bundling antara fixed broadband dan mobile.
      8. Saat ini telah hadir fitur baru untuk memantau akun IndiHome di aplikasi MyTelkomsel bagi Anda yang nomor Telkomselnya terdaftar pada profil IndiHome.
      9. Telepon rumah adalah salah satu layanan IndiHome yang dialihkan pengelolaannya ke Telkomsel. Tidak ada perubahan paket pada pelanggan.
      10. Mulai 1 Juli 2023 seluruh GraPARI Telkomsel sudah siap memberikan informasi dalam melayani pelanggan IndiHome.

      Dengan menggunakan pengetahuan yang Anda miliki tentang IndiHome dan Telkomsel, jawablah pertanyaan-pertanyaan tersebut dengan cara yang informatif dan mudah dipahami bagi pelanggan tidak harus terpaku dengan jawaban potensial untuk setiap pertanyaan. Jika ada pertanyaan yang tidak bisa dijawab dapat dialihkan ke admin.
      """

def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    combined_prompt = prompt + " " + question
    response = chat.send_message(combined_prompt, stream=True)
    return response

def ask_question(question):
    response = get_gemini_response(question, prompt)
    return response

def get_text_from_response(response):
   text = []
   for chunk in response:
    text.append(chunk.text)
   return ' '.join(text)