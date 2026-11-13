"""Truy xuất hai bước KHÔNG graph (kiểu Multi-Hop Dense Retrieval, không huấn luyện).

Hop 1: RRF(BM25, dense) như engine đã khóa.  Hop 2: với mỗi seed (top-n của hop 1) tạo truy vấn
"câu hỏi + nội dung seed", chạy lại BM25 và dense, rồi RRF với danh sách hop 1.
Không dùng entity linking, không dùng graph, không dùng nhãn/gold.
"""


def expansion_query(question, paper, mode):
    """mode='full': câu hỏi + title + abstract; mode='title': câu hỏi + title."""
    if mode == "full":
        return f"{question} {paper['title']}. {paper['abstract']}"
    if mode == "title":
        return f"{question} {paper['title']}"
    raise ValueError(f"mode không hợp lệ: {mode}")


VARIANTS = {  # khai báo TRƯỚC khi chạy; tw3_full là primary (3 seed, giống số seed của graph)
    "tw1_full": (1, "full"),
    "tw3_full": (3, "full"),
    "tw1_title": (1, "title"),
    "tw3_title": (3, "title"),
}
PRIMARY = "tw3_full"
