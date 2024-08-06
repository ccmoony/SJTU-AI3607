import jittor as jt
import jittor.nn as nn
rnn = nn.RNN(32, 128, 1)
input = jt.randn(32, 1024, 32)
output, hn = rnn(input)
print(output.shape, hn.shape)