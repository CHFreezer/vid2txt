# TODO

## ✅ 已修复: 停止转录按钮点击后 UI 不更新

**修复日期**: 2026-07-09
**修复**: `webui.py` — 删除 `stop_btn.click()` 中的 `cancels=[transcribe_event]`，让 generator 自然结束并完成 UI 更新。已通过 Playwright 实测验证。
