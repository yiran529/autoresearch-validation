# dense

Pinned base d2c38bc2062e06fdfd73185c508e7cc652933de6.

静态可构建；尚未进行 GPU 接入、数值一致性或完整训练验证。

- Public registration entrypoint is bridged; local tokenizer and completion evidence injected. GPU parity not yet tested.
- Original special-case hook replaced by shared registration; original byte estimates disabled. GPU parity not yet tested.
- Public registration bridged, completion fields added; accumulation fixed at 1. GPU parity not yet tested.
