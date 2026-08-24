# UI 字体资产

本文记录 bundled CJK fallback 的来源、用途与完整性；字体内容和许可分别以资产本身与 [`OFL.txt`](OFL.txt) 为准。

`NotoSansSC-Regular.otf` 是 Noto Sans CJK 2.004 的 Simplified Chinese subset，来源于官方
[`notofonts/noto-cjk`](https://github.com/notofonts/noto-cjk/tree/main/Sans/SubsetOTF/SC)。egui 把它安装为
最低优先级的 proportional/monospace fallback：默认 Latin 字体保持不变，CJK、箭头、标点和数学标签
不会显示为 missing-glyph box。

- SHA-256：`faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9`
- License：SIL Open Font License 1.1，见 [`OFL.txt`](OFL.txt)
- 上游发布说明：<https://github.com/notofonts/noto-cjk/blob/main/Sans/NEWS.md>
