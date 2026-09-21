import os
# Atur backend Qt ke xcb & matikan MIT-SHM agar kompatibel penuh dengan Wayland/Hyprland (mencegah black screen)
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_X11_NO_MITSHM"] = "1"

import cv2
import mediapipe as mp
import random
import time
import math

# Inisialisasi MediaPipe Hands standar untuk 1 TANGAN
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Kamus pola paten gestur [Jempol, Telunjuk, Tengah, Manis, Kelingking]
POLA_GESTUR = {
    (0, 1, 0, 0, 0): (1, "1 (Telunjuk)"),
    (0, 1, 1, 0, 0): (2, "2 (Telunjuk & Tengah)"),
    (0, 1, 1, 1, 0): (3, "3 (Telunjuk, Tengah, Manis)"),
    (0, 1, 1, 0, 1): (3, "3 (Telunjuk, Tengah, Kelingking)"),  # Variasi anatomis 3 jari
    (0, 1, 1, 1, 1): (4, "4 (Empat Jari)"),
    (1, 1, 1, 1, 1): (5, "5 (Semua 5 Jari)"),
    (1, 0, 0, 0, 0): (6, "6 (Hanya Jempol)"),
    (1, 1, 0, 0, 0): (7, "7 (Jempol & Telunjuk)"),
    (1, 1, 1, 0, 0): (8, "8 (Jempol, Telunjuk, Tengah)"),
    (1, 1, 1, 1, 0): (9, "9 (Semua Jari Kecuali Kelingking)"),
    (0, 0, 0, 0, 0): (0, "0 (Mengepal)"),
}

def dist(p1, p2):
    """Menghitung jarak Euclidean 2D antara dua landmark."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def buat_soal(prev_jawaban=None):
    """
    Membuat soal penjumlahan acak: angka1 + angka2.
    Jawaban kunci diambil dari angka satuan / paling belakang (modulus 10).
    """
    while True:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        hasil_asli = a + b
        jawaban_kunci = hasil_asli % 10
        if jawaban_kunci != prev_jawaban:
            return a, b, hasil_asli, jawaban_kunci

def deteksi_array_jari(hand_landmarks):
    """
    Mendeteksi status tiap jari dengan akurat:
    [Jempol, Telunjuk, Tengah, Manis, Kelingking]
    1 = berdiri / terbuka, 0 = ditekuk / mengepal
    """
    lm = hand_landmarks.landmark

    palm_size = dist(lm[0], lm[9])
    if palm_size == 0:
        palm_size = 0.001

    # 4 Jari utama: Telunjuk (8), Tengah (12), Manis (16), Kelingking (20)
    # Jari berdiri jika koordinat Y TIP lebih tinggi dari PIP dan jarak TIP ke Wrist lebih jauh dari PIP ke Wrist
    f_index = 1 if (lm[8].y < lm[6].y and dist(lm[8], lm[0]) > dist(lm[6], lm[0])) else 0
    f_middle = 1 if (lm[12].y < lm[10].y and dist(lm[12], lm[0]) > dist(lm[10], lm[0])) else 0
    f_ring = 1 if (lm[16].y < lm[14].y and dist(lm[16], lm[0]) > dist(lm[14], lm[0])) else 0
    f_pinky = 1 if (lm[20].y < lm[18].y and dist(lm[20], lm[0]) > dist(lm[18], lm[0])) else 0

    total_4_jari = f_index + f_middle + f_ring + f_pinky

    # Deteksi Jempol (Landmark 4: TIP, Landmark 3: IP, Landmark 2: MCP):
    d_thumb_index_base = dist(lm[4], lm[5]) / palm_size
    d_thumb_middle_base = dist(lm[4], lm[9]) / palm_size
    d_thumb_tip_pinky = dist(lm[4], lm[17])
    d_thumb_ip_pinky = dist(lm[3], lm[17])

    # 1. Thumbs-up murni (untuk angka 6):
    # Jempol harus mengacung tinggi ke atas MELEBIHI buku jari telunjuk (lm[5] dan lm[6]),
    # serta tulang jempol harus dalam keadaan lurus (bukan ditekuk di depan kepalan).
    thumbs_up = (
        total_4_jari == 0
        and lm[4].y < lm[5].y
        and lm[4].y < lm[6].y
        and dist(lm[4], lm[0]) > dist(lm[2], lm[0]) + 0.12 * palm_size
        and dist(lm[4], lm[2]) > 0.32 * palm_size
    )

    # 2. Jempol terbuka ke samping (untuk angka 5, 7, 8, 9):
    # Jempol merenggang keluar menjauhi pangkal telunjuk, tengah, dan kelingking.
    # Pada kepalan tangan (angka 0) atau jari 1-4, jempol menempel/menekuk dekat telapak, sehingga d_thumb_index_base kecil.
    thumb_spread = (
        d_thumb_index_base > 0.38
        and d_thumb_middle_base > 0.40
        and d_thumb_tip_pinky > d_thumb_ip_pinky
        and dist(lm[4], lm[2]) > 0.32 * palm_size
    )

    if thumbs_up or thumb_spread:
        f_thumb = 1
    else:
        f_thumb = 0

    return [f_thumb, f_index, f_middle, f_ring, f_pinky]

# State global UI untuk interaktivitas tombol
zoom_level = 1.0
is_fullscreen = False

# Definisi koordinat tombol (akan diupdate dinamis sesuai lebar frame w)
btn_zoom_in = [0, 0, 0, 0]
btn_zoom_out = [0, 0, 0, 0]
btn_fullscreen = [0, 0, 0, 0]
btn_minimize = [0, 0, 0, 0]

nama_jendela = "Math Gesture Game"

def on_mouse_click(event, x, y, flags, param):
    global zoom_level, is_fullscreen
    if event == cv2.EVENT_LBUTTONDOWN:
        # Klik Zoom In
        if btn_zoom_in[0] <= x <= btn_zoom_in[2] and btn_zoom_in[1] <= y <= btn_zoom_in[3]:
            zoom_level = min(2.5, round(zoom_level + 0.25, 2))
        # Klik Zoom Out (Mengecilkan)
        elif btn_zoom_out[0] <= x <= btn_zoom_out[2] and btn_zoom_out[1] <= y <= btn_zoom_out[3]:
            zoom_level = max(1.0, round(zoom_level - 0.25, 2))
        # Klik Fullscreen
        elif btn_fullscreen[0] <= x <= btn_fullscreen[2] and btn_fullscreen[1] <= y <= btn_fullscreen[3]:
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(nama_jendela, cv2.WND_PROP_FULLSCREEN, prop)
        # Klik Minimize (Kembali ke ukuran standar)
        elif btn_minimize[0] <= x <= btn_minimize[2] and btn_minimize[1] <= y <= btn_minimize[3]:
            is_fullscreen = False
            cv2.setWindowProperty(nama_jendela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(nama_jendela, 640, 480)
            zoom_level = 1.0

def main():
    global zoom_level, is_fullscreen, btn_zoom_in, btn_zoom_out, btn_fullscreen, btn_minimize

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera!")
        return

    # Gunakan WINDOW_NORMAL | WINDOW_GUI_NORMAL untuk menghapus toolbar default Qt yang tidak berguna
    cv2.namedWindow(nama_jendela, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(nama_jendela, 640, 480)
    cv2.setMouseCallback(nama_jendela, on_mouse_click)

    # Inisialisasi soal pertama
    angka1, angka2, hasil_asli, jawaban_kunci = buat_soal()
    skor = 0
    frame_benar_count = 0
    waktu_flash_benar = 0

    print("Game Matematika Deteksi Jari Siap Dimainkan!")
    print("Gunakan tombol klik di layar atau shortcut keyboard:")
    print("  'f': Fullscreen, 'm': Minimize/Reset, '+': Zoom In, '-': Zoom Out, 'q': Keluar")

    while True:
        success, img = cap.read()
        if not success:
            print("Gagal membaca frame kamera!")
            break

        # Mirror tampilan horizontal
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Terapkan Digital Zoom jika diaktifkan
        if zoom_level > 1.0:
            crop_w = int(w / zoom_level)
            crop_h = int(h / zoom_level)
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
            img = cv2.resize(img, (w, h))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Proses deteksi tangan dengan MediaPipe
        results = hands.process(img_rgb)

        pola_jari = None
        angka_terdeteksi = None
        nama_gestur = "Arahkan 1 tangan"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Gambar landmark & koneksi tangan
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Ekstrak array pola [J, T, T, M, K]
            pola_jari = deteksi_array_jari(hand_landmarks)
            pola_tuple = tuple(pola_jari)

            # Cocokkan dengan kamus pola paten
            if pola_tuple in POLA_GESTUR:
                angka_terdeteksi, nama_gestur = POLA_GESTUR[pola_tuple]
            else:
                angka_terdeteksi = None
                nama_gestur = f"Pola {pola_jari}"

        # Validasi Jawaban: Cek apakah angka_terdeteksi cocok dengan angka paling belakang
        if angka_terdeteksi is not None and angka_terdeteksi == jawaban_kunci:
            frame_benar_count += 1
            # Tahan stabil ~5 frame (~150ms) agar responsif dan langsung lanjut soal berikutnya
            if frame_benar_count >= 5:
                skor += 1
                waktu_flash_benar = time.time()
                # Langsung buat soal baru secara acak dan kontinu
                angka1, angka2, hasil_asli, jawaban_kunci = buat_soal(prev_jawaban=jawaban_kunci)
                frame_benar_count = 0
        else:
            frame_benar_count = 0

        sedang_benar = (time.time() - waktu_flash_benar) < 0.4

        # Update koordinat Quick Buttons di pojok kanan atas
        btn_zoom_in = [w - 390, 15, w - 305, 50]
        btn_zoom_out = [w - 295, 15, w - 210, 50]
        btn_fullscreen = [w - 200, 15, w - 105, 50]
        btn_minimize = [w - 95, 15, w - 15, 50]

        # Overlay Kotak HUD Transparan
        overlay = img.copy()
        # Box Soal Kiri Atas
        cv2.rectangle(overlay, (15, 15), (410, 145), (25, 25, 25), -1)
        # Box Status Kiri Bawah
        cv2.rectangle(overlay, (15, h - 130), (450, h - 15), (25, 25, 25), -1)

        # Background Quick Buttons Kanan Atas
        cv2.rectangle(overlay, (w - 400, 10), (w - 10, 55), (20, 20, 20), -1)

        # Gambar Tombol Quick Action
        cv2.rectangle(overlay, (btn_zoom_in[0], btn_zoom_in[1]), (btn_zoom_in[2], btn_zoom_in[3]), (50, 50, 50), -1)
        cv2.rectangle(overlay, (btn_zoom_out[0], btn_zoom_out[1]), (btn_zoom_out[2], btn_zoom_out[3]), (50, 50, 50), -1)
        cv2.rectangle(overlay, (btn_fullscreen[0], btn_fullscreen[1]), (btn_fullscreen[2], btn_fullscreen[3]), (50, 50, 50), -1)
        cv2.rectangle(overlay, (btn_minimize[0], btn_minimize[1]), (btn_minimize[2], btn_minimize[3]), (50, 50, 50), -1)

        img = cv2.addWeighted(overlay, 0.70, img, 0.30, 0)

        # Border Tombol Quick Action
        cv2.rectangle(img, (btn_zoom_in[0], btn_zoom_in[1]), (btn_zoom_in[2], btn_zoom_in[3]), (120, 120, 120), 1)
        cv2.rectangle(img, (btn_zoom_out[0], btn_zoom_out[1]), (btn_zoom_out[2], btn_zoom_out[3]), (120, 120, 120), 1)
        cv2.rectangle(img, (btn_fullscreen[0], btn_fullscreen[1]), (btn_fullscreen[2], btn_fullscreen[3]), (120, 120, 120), 1)
        cv2.rectangle(img, (btn_minimize[0], btn_minimize[1]), (btn_minimize[2], btn_minimize[3]), (120, 120, 120), 1)

        # Teks Tombol Quick Action
        zoom_text = f"+ Zoom" if zoom_level == 1.0 else f"+ Z({zoom_level}x)"
        cv2.putText(img, zoom_text, (btn_zoom_in[0] + 6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(img, "- Zoom", (btn_zoom_out[0] + 12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, "FullScr", (btn_fullscreen[0] + 15, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(img, "Minimize", (btn_minimize[0] + 8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Render Informasi Soal & Skor (TANPA membocorkan hasil asli)
        cv2.putText(img, f"Soal: {angka1} + {angka2} = ?", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)
        cv2.putText(img, "Tebak angka satuan (angka belakang)", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(img, f"Skor: {skor}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        # Render Deteksi Jari & Status (Tetap menampilkan angka jari secara jelas)
        if pola_jari is not None:
            teks_pola = f"Pola [J,T,T,M,K]: {pola_jari}"
        else:
            teks_pola = "Pola [J,T,T,M,K]: -"

        cv2.putText(img, teks_pola, (30, h - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 255), 1)

        # Tampilkan secara gamblang angka yang ditunjukkan jari
        if angka_terdeteksi is not None:
            teks_angka = f"Jari Menunjukkan Angka: {angka_terdeteksi}"
            warna_angka = (0, 255, 0) if angka_terdeteksi == jawaban_kunci else (0, 255, 255)
        else:
            teks_angka = f"Jari: {nama_gestur}"
            warna_angka = (180, 180, 180)

        cv2.putText(img, teks_angka, (30, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, warna_angka, 2)

        # Status Permainan (TANPA membocorkan target jawaban)
        if sedang_benar:
            status_str = "BENAR! (+1)"
            warna_status = (0, 255, 0)
        elif pola_jari is None:
            status_str = "Tunjukkan 1 tangan!"
            warna_status = (180, 180, 180)
        else:
            status_str = "Ayo Jawab!"
            warna_status = (0, 165, 255)

        cv2.putText(img, status_str, (30, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, warna_status, 2)

        # Efek Banner Flash di Tengah Layar saat Jawaban Benar
        if sedang_benar:
            cv2.putText(img, "BENAR!", (w // 2 - 110, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        cv2.imshow(nama_jendela, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):  # Shortcut Fullscreen
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(nama_jendela, cv2.WND_PROP_FULLSCREEN, prop)
        elif key in [ord('+'), ord('=')]:  # Shortcut Zoom In
            zoom_level = min(2.5, round(zoom_level + 0.25, 2))
        elif key in [ord('-'), ord('_')]:  # Shortcut Zoom Out
            zoom_level = max(1.0, round(zoom_level - 0.25, 2))
        elif key == ord('m'):  # Shortcut Minimize
            is_fullscreen = False
            cv2.setWindowProperty(nama_jendela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(nama_jendela, 640, 480)
            zoom_level = 1.0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()