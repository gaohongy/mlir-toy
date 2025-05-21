## 对于 op 进行扩展的几点内容

1. 自定义 IR 输出格式 assemblyFormat   ✅
2. 为 Op 添加固定函数和自定义函数   hasVerifier, getBitWidth ✅
3. 使用新的 Trait，需要给出具体实现 ✅

添加这 Op 似乎是有固定用途，不知道是否需要单独分一个 ex
4. 添加一个 ReturnOp                    ✅
5. 添加 FuncOp，这里面涉及的条目有些繁多
6. 添加 CallOp
