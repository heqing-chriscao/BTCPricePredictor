# LSTM-Based Stock Price Prediction
This repository contains the code， data, and documentation for our Bitcoin prediction models. The project explores different model architectures and features.

## Runnable Scripts/Commands
All the code in this repository is provided as Jupyter notebooks (`.ipynb`) that can be run directly on Google Colab. Below are the steps to reproduce the results:

1. **Open the .ipynb file in colab**:
   - Open your browser and go to [Google Colab](https://colab.research.google.com).
   - Upload the desired `.ipynb` file to Colab or use the GitHub import option in Colab to open the file directly from this repository.

2. **Load Data**:
   - Copy the `data` folder from this repository to your Google Drive.
   - Mount the google drive folders to get the data:
     ```python
     from google.colab import drive
     drive.mount("/content/drive")
     ```
     
3. **Data Preprocessing**:
   We extract the desired data columns, normalize the data, and split the feature datasets.
   ```
   def data_preprocessing_with_closed_price_and_min_max_scaler(data, seq_length=60):
      train_ratio = 0.7
      val_ratio = 0.15
      close_data = data["Close"].values.reshape(-1, 1)
      scaler = MinMaxScaler(feature_range=(0, 10))
      data_scaled = scaler.fit_transform(close_data)
      data_scaled = pd.DataFrame(data_scaled, columns=["Close"])
  
      def create_feature_datasets(data, x_size):
          x_datasets = []
          y_datasets = []
          for i in range(len(data) - x_size):
              x = data[i:i + x_size].values.reshape(-1, 1)
              y = data[i + x_size:i + x_size + 1].values.reshape(1)
              x_datasets.append(x)
              y_datasets.append(y)
          X = np.array(x_datasets)
          y = np.array(y_datasets)
          X_train =  X[:int(len(X) * train_ratio)]
          y_train = y[:int(len(y) * train_ratio)]
          X_val = X[int(len(X) * train_ratio):int(len(X) * (train_ratio + val_ratio))]
          y_val = y[int(len(y) * train_ratio):int(len(y) * (train_ratio + val_ratio))]
          X_test = X[int(len(X) * (train_ratio + val_ratio)):]
          y_test = y[int(len(y) * (train_ratio + val_ratio)):]
  
          indices = np.arange(X_train.shape[0])
          np.random.shuffle(indices)
          X_train_shuffled = X_train[indices]
          y_train_shuffled = y_train[indices]
  
          return torch.Tensor(X_train_shuffled).to(device), torch.Tensor(X_val).to(device), torch.Tensor(X_test).to(device), torch.Tensor(y_train_shuffled).to(device), torch.Tensor(y_val).to(device), torch.Tensor(y_test).to(device)
  
      return create_feature_datasets(data_scaled, seq_length), scaler
  
    (X_train, X_val, X_test, y_train, y_val, y_test), min_max_scaler = data_preprocessing_with_closed_price_and_min_max_scaler(data, 60)
    ```

4. **Data Preprocessing**:
  - LSTM-based Models:
    ```
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Initial hidden state
            c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Initial cell state
            out, _ = self.lstm(x, (h_0, c_0))
            out = self.fc(out[:, -1, :])  # Get the output from the last time step
            return out
    ```
  - Transformer-based Models:
    ```
    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super(TransformerModel, self).__init__()
            self.hidden_size = hidden_size
    
            # Input embedding to project input_size to hidden_size
            self.input_proj = nn.Linear(input_size, hidden_size)
    
            # Positional encoding for sequence information
            self.positional_encoding = PositionalEncoding(hidden_size, dropout)
    
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
            # Fully connected output layer
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            # x shape: (batch_size, seq_length, input_size)
    
            # Project input to the hidden_size
            x = self.input_proj(x)
    
            # Apply positional encoding
            x = self.positional_encoding(x)
    
            # Transformer expects input in (seq_length, batch_size, hidden_size) format
            x = x.permute(1, 0, 2)  # Change to (seq_length, batch_size, hidden_size)
    
            # Transformer Encoder
            x = self.transformer_encoder(x)
    
            # Get the output from the last time step
            x = x[-1, :, :]  # Last time step, shape: (batch_size, hidden_size)
    
            # Fully connected layer
            x = self.fc(x)
            return x
    
    
    class PositionalEncoding(nn.Module):
        def __init__(self, hidden_size, dropout=0.1, max_len=100):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
    
            # Create positional encoding matrix
            pe = torch.zeros(max_len, hidden_size)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Add batch dimension
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            # Add positional encoding to the input
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    ```
5. **Model Training**:
   - Loss function:
     ```
     class RMSELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
    
        def forward(self,yhat,y):
            return torch.sqrt(self.mse(yhat,y))
     ```
   - Training:
     ```
      dataset = TensorDataset(X_train, y_train)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
      
      # Model, loss, optimizer
      model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
      model = model.to(device)
      criterion = RMSELoss()
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
      
      epoch_losses = []
      
      # Training loop with batching
      for epoch in range(num_epochs):
          model.train()
          for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
              optimizer.zero_grad()
              outputs = model(batch_x)
              loss = criterion(outputs, batch_y)
              loss.backward()
              optimizer.step()
      
          epoch_losses.append(loss.item())
          if (epoch + 1) % 5 == 0:
              print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
     ```

6. **Data Visualization**：
   - Prediction v.s. GT:
     ```
      tensor1 = predictions_reversed.view(-1)
      tensor2 = y_val_reversed.view(-1)
      
      # Plotting the line chart
      plt.figure(figsize=(8, 5))
      plt.plot(tensor1.numpy(), label='Predictions', marker='x')
      plt.plot(tensor2.numpy(), label='GT', marker='.')
      plt.title('Predictions vs GT on Validation Set')
      plt.xlabel('Index')
      plt.ylabel('Value')
      plt.legend()
      plt.grid(True)
      plt.show()
     ```

## Contribution
- Mengyan Cao:
- Hoiting Mok:
- Jiaxuan Li:
- Ruolong Mao:
   - LSTM_RMSE_Baseline.ipynb
   - LSTM_RMSE_Close_Price.ipynb
   - LSTM_RMSE_Five_Features.ipynb
   - LSTM_RMSE_Close_Price+spy.ipynb
   - data/BTC-USD_stock_data.csv
   - data/BTC-USD_stock_data_spy.csv
   - Tuner For Model/Tuner of LSTM_RMSE_Close_Price.ipynb
