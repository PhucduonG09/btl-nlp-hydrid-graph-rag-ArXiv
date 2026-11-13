"""Thứ tự đưa nguồn vào ngữ cảnh LLM (thăm dò 'lost in the middle'). Hàm thuần, không gọi model."""

MODES = ("rank", "reverse", "sandwich", "top5")


def reorder(ranking, mode):
    """ranking: danh sách paper_id theo hạng (tốt nhất trước). Trả về thứ tự nguồn đưa vào prompt.

    rank     giữ nguyên hạng (baseline hiện tại)
    reverse  tốt nhất ở CUỐI (kiểm tra: model có thiên về nguồn đầu hay không)
    sandwich hạng 1,3,5.. ở đầu và hạng 2,4,6.. ở cuối theo thứ tự ngược: các hạng cao nằm ở hai rìa, hạng thấp ở giữa
             (cách giảm 'lost in the middle' thường được đề xuất); với top-10: 1,3,5,7,9,10,8,6,4,2
    top5     chỉ 5 nguồn đầu theo hạng (ngữ cảnh ngắn hơn, proxy cho nén bằng cắt)
    """
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate candidates")
    if mode == "rank":
        return list(ranking)
    if mode == "reverse":
        return list(ranking)[::-1]
    if mode == "sandwich":
        return list(ranking[0::2]) + list(ranking[1::2])[::-1]
    if mode == "top5":
        return list(ranking[:5])
    raise ValueError(f"mode không hợp lệ: {mode}")
