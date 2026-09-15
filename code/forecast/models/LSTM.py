import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    简洁有效的LSTM模型用于时间序列预测
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention if hasattr(configs, 'output_attention') else False
        
        # 模型参数
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # 主LSTM层 - 使用简单但有效的配置
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # 单向LSTM
        )
        
        # 简单的投影层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 使用序列到序列的线性变换，类似于DLinear的做法
            self.projection = nn.Linear(self.hidden_size, self.c_out)
            # 时间维度投影：从seq_len投影到pred_len
            self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(self.hidden_size, self.c_out)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(self.hidden_size, self.c_out)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(self.dropout)
            self.projection = nn.Linear(self.hidden_size * self.seq_len, configs.num_class)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        统一的前向传播方法
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
    
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        简洁的预测方法 - 模仿TimesNet的标准化策略
        """
        # 数据标准化 - 使用与TimesNet相同的策略
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x_enc)  # [B, T, hidden_size]
        
        # 特征投影
        lstm_out = self.projection(lstm_out)  # [B, T, c_out]
        
        # 时间维度变换：从seq_len到pred_len
        # 转置以便对时间维度进行线性变换
        lstm_out = lstm_out.permute(0, 2, 1)  # [B, c_out, T]
        predictions = self.temporal_projection(lstm_out)  # [B, c_out, pred_len]
        predictions = predictions.permute(0, 2, 1)  # [B, pred_len, c_out]
        
        # 反标准化
        predictions = predictions * stdev
        predictions = predictions + means
        
        return predictions
    
    def imputation(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        填充缺失值
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x_enc)
        
        # 投影到输出维度
        imputed_values = self.projection(lstm_out)
        
        return imputed_values
    
    def anomaly_detection(self, x_enc):
        """
        异常检测
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x_enc)
        
        # 投影到输出维度
        anomaly_score = self.projection(lstm_out)
        
        return anomaly_score
    
    def classification(self, x_enc, x_mark_enc=None):
        """
        分类任务
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x_enc)
        
        # 全局平均池化
        output = lstm_out.mean(dim=1)  # [B, hidden_size]
        
        # 分类
        output = self.act(output)
        output = self.dropout_layer(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        
        return output 