from __future__ import print_function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling1D
import args

class FTModel:
    def __init__(self, max_features):
        self.max_features = max_features
        self.model = self.model_builder()

    def model_builder(self):
        model = Sequential()
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        # 词汇索引映射到词向量
        model.add(Embedding(self.max_features,
                            args.embedding_dims,
                            input_length=args.maxlen))
        # 平均计算文档中所有词汇的的词嵌入
        model.add(GlobalAveragePooling1D())

        #投射到输出层，sigmoid压缩
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

        model.summary()

        return model
