from fpdf import FPDF
from datetime import datetime
import os
from bs4 import BeautifulSoup

class PDFReport:
    def generate(self, disease_name, confidence, report_text, image_pil, file_path):
        """
        يولّد PDF من HTML-like report_text ويخزنه في file_path
        """

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()  # ✅ مهم جداً

        # عرض الصفحة المرن
        page_width = pdf.w - 2 * pdf.l_margin
        col1 = page_width * 0.3
        col2 = page_width * 0.7

        # 🟢 العنوان
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 10, "Plant Disease Diagnosis Report", ln=True, align='C')
        pdf.ln(5)

        # 🟢 إضافة الصورة
        if image_pil:
            temp_img = "temp_pdf_image.jpg"
            image_pil.convert("RGB").save(temp_img, "JPEG")
            pdf.image(temp_img, x=40, w=120)
            pdf.ln(10)
        else:
            temp_img = None

        # 🟢 جدول المعلومات
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(230, 240, 230)
        pdf.cell(col1, 10, "Feature", border=1, fill=True)
        pdf.cell(col2, 10, "Diagnosis Details", border=1, fill=True, ln=True)

        pdf.set_font("Arial", '', 12)
        pdf.cell(col1, 10, "Disease Name", border=1)
        pdf.cell(col2, 10, str(disease_name), border=1, ln=True)

        pdf.cell(col1, 10, "Confidence Level", border=1)
        pdf.cell(col2, 10, f"{confidence:.2f}%", border=1, ln=True)

        pdf.cell(col1, 10, "Date", border=1)
        pdf.cell(col2, 10, datetime.now().strftime("%Y-%m-%d %H:%M"), border=1, ln=True)

        pdf.ln(10)

        # 🟢 AI Recommendations
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "AI Expert Recommendations:", ln=True)
        pdf.ln(2)

        pdf.set_font("Arial", '', 11)

        soup = BeautifulSoup(report_text, "html.parser")

        for elem in soup.children:
            text = elem.get_text(strip=True) if hasattr(elem, "get_text") else str(elem).strip()

            # ✅ تجاهل النصوص الفارغة
            if not text:
                continue

            # ✅ تأكد أننا في بداية السطر
            pdf.set_x(pdf.l_margin)

            if elem.name == "p":
                pdf.multi_cell(0, 6, text)
                pdf.ln(1)

            elif elem.name in ["ul", "ol"]:
                for li in elem.find_all("li"):
                    li_text = li.get_text(strip=True)
                    if li_text:
                        pdf.set_x(pdf.l_margin)
                        pdf.multi_cell(0, 6, f"- {li_text}")
                pdf.ln(2)

            elif elem.name and elem.name.startswith("h"):
                pdf.set_font("Arial", 'B', 12)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 7, text)
                pdf.set_font("Arial", '', 11)
                pdf.ln(1)

            else:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 6, text)
                pdf.ln(1)

        # 🧹 حذف الصورة المؤقتة
        if temp_img and os.path.exists(temp_img):
            os.remove(temp_img)

        # 📁 حفظ الملف
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pdf.output(file_path)

        return file_path


pdf_service = PDFReport()
