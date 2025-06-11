# Giải thích chi tiết Notebook huấn luyện mô hình CNN

Notebook này trình bày quá trình xây dựng, huấn luyện và đánh giá một mô hình Mạng nơ-ron tích chập (CNN) để phân loại hình ảnh.

## 1. Các hàm (Functions) được định nghĩa

Dưới đây là các hàm chính được định nghĩa trong notebook, cùng với mục đích, input, output và cách hoạt động nội bộ của chúng.

### `load_data(data_path)`

* **Mục đích**: Tải dữ liệu hình ảnh từ một thư mục được chỉ định và chuẩn bị chúng cho quá trình huấn luyện mô hình.
* **Input**:
    * `data_path` (string): Đường dẫn đến thư mục chứa dữ liệu hình ảnh.
* **Output**:
    * `train_ds` (tf.data.Dataset): Dataset huấn luyện.
    * `val_ds` (tf.data.Dataset): Dataset kiểm tra (validation).
    * `test_ds` (tf.data.Dataset): Dataset kiểm tra cuối cùng (nếu có tách riêng).
    * `class_names` (list): Danh sách tên các lớp (nhãn) trong dataset.
* **Cách hoạt động nội bộ**:
    1.  Sử dụng `tf.keras.utils.image_dataset_from_directory` để tải hình ảnh. Hàm này tự động suy luận các lớp từ cấu trúc thư mục con và tạo các dataset `tf.data.Dataset`.
    2.  Thiết lập `image_size` và `batch_size` cho quá trình tải dữ liệu.
    3.  Tách dữ liệu thành các tập huấn luyện, kiểm tra (validation) và kiểm tra cuối cùng (test) dựa trên tỷ lệ `validation_split` và `subset`.
    4.  Sử dụng `cache()`, `shuffle()`, `prefetch()` để tối ưu hóa hiệu suất đọc dữ liệu.

### `data_augmentation_layer()`

* **Mục đích**: Tạo một lớp tăng cường dữ liệu (data augmentation) để áp dụng các biến đổi ngẫu nhiên lên hình ảnh, giúp mô hình học được các đặc trưng mạnh mẽ hơn và giảm thiểu overfitting.
* **Input**: Không có.
* **Output**:
    * `tf.keras.Sequential` layer: Một lớp tuần tự chứa các phép biến đổi tăng cường dữ liệu.
* **Cách hoạt động nội bộ**:
    1.  Tạo một `tf.keras.Sequential` object.
    2.  Thêm các lớp tăng cường dữ liệu như `RandomFlip("horizontal")` (lật ngang ngẫu nhiên) và `RandomRotation(0.1)` (xoay ngẫu nhiên một góc nhỏ). Có thể thêm các lớp khác như `RandomZoom`, `RandomContrast` tùy theo nhu cầu.

### `build_model(num_classes, input_shape)`

* **Mục đích**: Xây dựng kiến trúc mô hình CNN.
* **Input**:
    * `num_classes` (int): Số lượng lớp đầu ra (số lượng nhãn phân loại).
    * `input_shape` (tuple): Kích thước của ảnh đầu vào (ví dụ: `(256, 256, 3)` cho ảnh màu 256x256).
* **Output**:
    * `tf.keras.Model`: Mô hình CNN đã được biên dịch.
* **Cách hoạt động nội bộ**:
    1.  Khởi tạo `tf.keras.Sequential` làm backbone của mô hình.
    2.  **Lớp tiền xử lý**:
        * `Rescaling(1./255)`: Chuẩn hóa giá trị pixel từ `[0, 255]` về `[0, 1]`.
    3.  **Lớp tăng cường dữ liệu (tùy chọn)**:
        * `data_augmentation_layer()`: Áp dụng các biến đổi tăng cường dữ liệu trong quá trình huấn luyện. Điều này giúp mô hình không bao giờ thấy chính xác cùng một hình ảnh hai lần, giúp cải thiện khả năng tổng quát hóa.
    4.  **Các lớp tích chập (Convolutional Layers)**:
        * `Conv2D` (Convolutional 2D): Áp dụng các bộ lọc (kernel) để trích xuất các đặc trưng từ hình ảnh. Mỗi lớp `Conv2D` có số lượng bộ lọc khác nhau (ví dụ: 32, 64, 128) và kích thước kernel (ví dụ: `(3, 3)`).
        * `Activation('relu')`: Hàm kích hoạt ReLU (Rectified Linear Unit) được sử dụng sau mỗi lớp tích chập để thêm tính phi tuyến vào mô hình.
        * `MaxPooling2D` (Max Pooling 2D): Giảm chiều không gian của đầu ra từ lớp tích chập, giúp giảm số lượng tham số và tính toán, đồng thời giúp mô hình trở nên bất biến với các dịch chuyển nhỏ trong hình ảnh.
    5.  **Lớp Flatten**:
        * `Flatten()`: Chuyển đổi đầu ra từ các lớp tích chập (có dạng 2D hoặc 3D) thành một vector 1D, sẵn sàng cho các lớp Dense.
    6.  **Các lớp Dense (Fully Connected Layers)**:
        * `Dense` (Fully Connected): Các lớp nơ-ron truyền thống, nơi mỗi nơ-ron được kết nối với tất cả các nơ-ron của lớp trước.
        * `Dropout(0.5)`: Ngẫu nhiên bỏ qua một số nơ-ron trong quá trình huấn luyện để ngăn chặn overfitting.
    7.  **Lớp đầu ra**:
        * `Dense(num_classes)`: Lớp đầu ra với số nơ-ron bằng số lượng lớp cần phân loại.
        * `Activation('softmax')`: Hàm kích hoạt Softmax được sử dụng cho bài toán phân loại đa lớp, tạo ra phân phối xác suất cho mỗi lớp. (Nếu là bài toán phân loại nhị phân, có thể sử dụng `sigmoid`).
    8.  **Biên dịch mô hình**:
        * `model.compile()`: Cấu hình mô hình cho quá trình huấn luyện.
            * `optimizer='adam'`: Thuật toán tối ưu hóa Adam được sử dụng để điều chỉnh trọng số của mô hình.
            * `loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`: Hàm mất mát cho bài toán phân loại. `from_logits=True` có nghĩa là đầu ra của lớp cuối cùng chưa được chuẩn hóa qua hàm Softmax.
            * `metrics=['accuracy']`: Các chỉ số được sử dụng để đánh giá hiệu suất của mô hình trong quá trình huấn luyện và đánh giá.

### `train_model(model, train_ds, val_ds, epochs)`

* **Mục đích**: Huấn luyện mô hình CNN đã được xây dựng bằng cách lặp lại trên tập dữ liệu huấn luyện và điều chỉnh trọng số của mô hình.
* **Input**:
    * `model` (tf.keras.Model): Mô hình CNN đã được biên dịch.
    * `train_ds` (tf.data.Dataset): Dataset huấn luyện.
    * `val_ds` (tf.data.Dataset): Dataset kiểm tra (validation).
    * `epochs` (int): Số kỷ nguyên (epochs) để huấn luyện mô hình.
* **Output**:
    * `tf.keras.callbacks.History`: Đối tượng History chứa thông tin về quá trình huấn luyện (mất mát, độ chính xác trên tập huấn luyện và kiểm tra).
* **Cách hoạt động nội bộ**:
    1.  Sử dụng `model.fit()` để bắt đầu quá trình huấn luyện.
    2.  Trong mỗi kỷ nguyên, mô hình sẽ đi qua toàn bộ tập huấn luyện, tính toán mất mát và cập nhật trọng số.
    3.  Hiệu suất của mô hình cũng được đánh giá trên tập kiểm tra (validation set) sau mỗi kỷ nguyên.

### `evaluate_model(model, test_ds)`

* **Mục đích**: Đánh giá hiệu suất của mô hình đã huấn luyện trên tập dữ liệu kiểm tra độc lập.
* **Input**:
    * `model` (tf.keras.Model): Mô hình CNN đã huấn luyện.
    * `test_ds` (tf.data.Dataset): Dataset kiểm tra cuối cùng.
* **Output**:
    * Kết quả đánh giá (mất mát, độ chính xác).
* **Cách hoạt động nội bộ**:
    1.  Sử dụng `model.evaluate()` để tính toán mất mát và các chỉ số (ví dụ: độ chính xác) trên tập dữ liệu kiểm tra.

### `plot_training_history(history)`

* **Mục đích**: Vẽ biểu đồ hiển thị quá trình huấn luyện của mô hình, bao gồm mất mát và độ chính xác trên cả tập huấn luyện và kiểm tra.
* **Input**:
    * `history` (tf.keras.callbacks.History): Đối tượng History trả về từ hàm `train_model`.
* **Output**:
    * Hiển thị biểu đồ.
* **Cách hoạt động nội bộ**:
    1.  Trích xuất dữ liệu mất mát và độ chính xác từ đối tượng `history`.
    2.  Sử dụng Matplotlib để vẽ hai biểu đồ: một cho mất mát (training loss và validation loss) và một cho độ chính xác (training accuracy và validation accuracy) theo từng kỷ nguyên.

### `run_training(params)`

* **Mục đích**: Hàm tổng hợp toàn bộ quy trình huấn luyện mô hình, từ tải dữ liệu, xây dựng mô hình, huấn luyện, đánh giá và hiển thị kết quả.
* **Input**:
    * `params` (dict): Một dictionary chứa các tham số huấn luyện như `data_path`, `epochs`, `batch_size`, `image_size`.
* **Output**: Không có (nhưng sẽ hiển thị kết quả và biểu đồ).
* **Cách hoạt động nội bộ**:
    1.  Giải nén các tham số từ dictionary `params`.
    2.  Gọi `load_data()` để tải và chuẩn bị dữ liệu.
    3.  Gọi `build_model()` để xây dựng mô hình CNN.
    4.  Gọi `train_model()` để huấn luyện mô hình.
    5.  Gọi `evaluate_model()` để đánh giá mô hình trên tập kiểm tra.
    6.  Gọi `plot_training_history()` để hiển thị biểu đồ quá trình huấn luyện.

## 2. Giải thích toàn bộ Workflow của Notebook

Workflow của notebook này được tổ chức thành các bước rõ ràng:

### 2.1. Chuẩn bị dữ liệu

1.  **Tải dữ liệu**: Dữ liệu hình ảnh được tải từ một thư mục được chỉ định (`data_path`). Notebook sử dụng `tf.keras.utils.image_dataset_from_directory`, một công cụ tiện lợi của TensorFlow để tự động nhận diện các lớp từ tên thư mục con và tải hình ảnh.
2.  **Tách tập dữ liệu**: Dữ liệu được chia thành ba tập:
    * **Tập huấn luyện (Training Set)**: Được sử dụng để huấn luyện mô hình, nơi mô hình học cách ánh xạ đầu vào đến đầu ra.
    * **Tập kiểm tra (Validation Set)**: Được sử dụng để theo dõi hiệu suất của mô hình trong quá trình huấn luyện và điều chỉnh các siêu tham số. Điều này giúp phát hiện và ngăn chặn overfitting.
    * **Tập kiểm tra cuối cùng (Test Set)**: Một tập dữ liệu hoàn toàn độc lập, chưa từng được mô hình nhìn thấy trong quá trình huấn luyện hoặc điều chỉnh siêu tham số. Nó được sử dụng để đánh giá hiệu suất tổng quát của mô hình sau khi huấn luyện.
3.  **Tăng cường dữ liệu (Data Augmentation)**: Các kỹ thuật tăng cường dữ liệu như lật ngang ngẫu nhiên (`RandomFlip`) và xoay ngẫu nhiên (`RandomRotation`) được áp dụng. Điều này giúp tạo ra các phiên bản biến thể của hình ảnh huấn luyện, mở rộng dataset một cách nhân tạo và giúp mô hình trở nên mạnh mẽ hơn với các biến thể nhỏ trong dữ liệu thực tế, giảm thiểu overfitting.
4.  **Tiền xử lý (Preprocessing)**: Hình ảnh được chuẩn hóa giá trị pixel về khoảng `[0, 1]` bằng cách chia cho 255 (`Rescaling(1./255)`). Điều này là quan trọng vì các mạng nơ-ron thường hoạt động tốt hơn với các giá trị đầu vào nhỏ và chuẩn hóa.
5.  **Tối ưu hóa hiệu suất đọc dữ liệu**: Các phương thức như `cache()`, `shuffle()`, và `prefetch()` được sử dụng để tối ưu hóa việc nạp dữ liệu vào mô hình, giúp quá trình huấn luyện diễn ra nhanh hơn và hiệu quả hơn.

### 2.2. Huấn luyện mô hình

1.  **Xây dựng mô hình**: Mô hình CNN được xây dựng bằng cách sử dụng `tf.keras.Sequential`. Kiến trúc bao gồm nhiều lớp tích chập (`Conv2D`) và lớp gộp (`MaxPooling2D`) để trích xuất các đặc trưng phân cấp từ hình ảnh. Sau đó, đầu ra được làm phẳng (`Flatten`) và đưa qua các lớp kết nối đầy đủ (`Dense`) để phân loại. Lớp `Dropout` được thêm vào để ngăn chặn overfitting.
2.  **Biên dịch mô hình**: Mô hình được biên dịch với một thuật toán tối ưu hóa (Adam), hàm mất mát (SparseCategoricalCrossentropy) và các chỉ số đánh giá (accuracy).
3.  **Bắt đầu huấn luyện**: Hàm `train_model` được gọi để thực hiện quá trình huấn luyện. Mô hình lặp lại trên tập huấn luyện trong một số kỷ nguyên (`epochs` ) đã định. Trong mỗi kỷ nguyên, mô hình học từ dữ liệu, điều chỉnh trọng số để giảm thiểu hàm mất mát, và sau đó được đánh giá trên tập kiểm tra (validation set) để theo dõi hiệu suất.

### 2.3. Đánh giá mô hình

1.  **Đánh giá trên tập kiểm tra**: Sau khi quá trình huấn luyện hoàn tất, mô hình được đánh giá trên tập kiểm tra cuối cùng (`test_ds`) bằng hàm `evaluate_model`. Tập kiểm tra này là độc lập và không được sử dụng trong quá trình huấn luyện hoặc điều chỉnh, cung cấp một ước lượng công bằng về khả năng tổng quát hóa của mô hình trên dữ liệu chưa từng thấy.
2.  **Trực quan hóa kết quả**: Hàm `plot_training_history` được sử dụng để hiển thị biểu đồ về mất mát và độ chính xác của cả tập huấn luyện và kiểm tra qua từng kỷ nguyên. Điều này giúp dễ dàng nhận biết các vấn đề như overfitting (khi độ chính xác trên tập huấn luyện tiếp tục tăng nhưng độ chính xác trên tập kiểm tra bắt đầu giảm) hoặc underfitting (khi độ chính xác trên cả hai tập đều thấp).

### 2.4. Lưu và sử dụng mô hình sau huấn luyện

* Notebook này **chưa có bước rõ ràng để lưu mô hình** sau khi huấn luyện. Tuy nhiên, sau khi huấn luyện xong, người dùng có thể dễ dàng thêm dòng code để lưu mô hình bằng cách sử dụng `model.save('my_cnn_model')` hoặc `model.save_weights('my_cnn_weights.h5')`.
* Để sử dụng mô hình đã huấn luyện, người dùng sẽ cần tải mô hình đã lưu (`tf.keras.models.load_model('my_cnn_model')`) và sau đó sử dụng phương thức `model.predict()` để đưa ra dự đoán trên dữ liệu mới.

## 3. Kiến trúc CNN đã được xây dựng

Kiến trúc CNN trong notebook này theo một mô hình tuần tự (Sequential) và bao gồm các thành phần sau:

1.  **`Rescaling(1./255)`**:
    * **Vai trò**: Chuẩn hóa giá trị pixel của hình ảnh đầu vào từ thang `[0, 255]` về thang `[0, 1]`.
    * **Giải thích**: Điều này giúp quá trình huấn luyện mô hình ổn định và hiệu quả hơn vì các thuật toán tối ưu hóa thường hoạt động tốt hơn với các giá trị đầu vào nhỏ.
2.  **`data_augmentation_layer()`** (ví dụ: `RandomFlip("horizontal")`, `RandomRotation(0.1)`):
    * **Vai trò**: Tăng cường dữ liệu bằng cách áp dụng các biến đổi ngẫu nhiên (lật, xoay) lên hình ảnh huấn luyện trong mỗi epoch.
    * **Giải thích**: Giúp mô hình học được các đặc trưng mạnh mẽ hơn và giảm thiểu overfitting bằng cách giới thiệu các biến thể của dữ liệu gốc.
3.  **`Conv2D(32, (3, 3), activation='relu')`**:
    * **Vai trò**: Lớp tích chập đầu tiên. Áp dụng 32 bộ lọc (kernel) có kích thước 3x3 để trích xuất các đặc trưng cấp thấp từ hình ảnh, như các cạnh và góc. Hàm kích hoạt ReLU được sử dụng để thêm tính phi tuyến.
4.  **`MaxPooling2D()`**:
    * **Vai trò**: Giảm chiều không gian của đầu ra từ lớp tích chập trước.
    * **Giải thích**: Giúp giảm số lượng tham số và tính toán, đồng thời làm cho mô hình trở nên bất biến với các dịch chuyển nhỏ của đặc trưng trong hình ảnh.
5.  **`Conv2D(64, (3, 3), activation='relu')`**:
    * **Vai trò**: Lớp tích chập thứ hai với 64 bộ lọc 3x3. Học các đặc trưng phức tạp hơn dựa trên đầu ra của lớp tích chập trước.
6.  **`MaxPooling2D()`**:
    * **Vai trò**: Tương tự như lớp gộp trước, tiếp tục giảm chiều không gian.
7.  **`Conv2D(128, (3, 3), activation='relu')`**:
    * **Vai trò**: Lớp tích chập thứ ba với 128 bộ lọc 3x3. Học các đặc trưng trừu tượng và phức tạp hơn nữa.
8.  **`MaxPooling2D()`**:
    * **Vai trò**: Tiếp tục giảm chiều không gian.
9.  **`Flatten()`**:
    * **Vai trò**: Chuyển đổi đầu ra 3D (chiều cao, chiều rộng, kênh) từ các lớp tích chập thành một vector 1D.
    * **Giải thích**: Điều này là cần thiết để kết nối với các lớp Dense (fully connected) tiếp theo.
10. **`Dense(128, activation='relu')`**:
    * **Vai trò**: Lớp kết nối đầy đủ đầu tiên với 128 nơ-ron. Học các mối quan hệ phi tuyến giữa các đặc trưng đã trích xuất.
11. **`Dropout(0.5)`**:
    * **Vai trò**: Ngẫu nhiên bỏ qua 50% số nơ-ron trong quá trình huấn luyện.
    * **Giải thích**: Kỹ thuật điều hòa này giúp ngăn chặn overfitting bằng cách buộc mô hình phải học các biểu diễn mạnh mẽ hơn và ít phụ thuộc vào các nơ-ron cụ thể.
12. **`Dense(num_classes, activation='softmax')`**:
    * **Vai trò**: Lớp đầu ra với số lượng nơ-ron bằng số lượng lớp cần phân loại. Hàm kích hoạt Softmax tạo ra phân phối xác suất trên các lớp, với tổng các xác suất bằng 1.
    * **Giải thích**: Đầu ra này biểu thị xác suất của hình ảnh đầu vào thuộc về mỗi lớp.

Kiến trúc này là một cấu trúc CNN tiêu chuẩn, bắt đầu bằng các lớp tích chập và gộp để trích xuất đặc trưng, sau đó sử dụng các lớp Dense để phân loại cuối cùng.

## 4. Dataset cụ thể và Kỹ thuật Augmentation, Preprocessing

### 4.1. Dataset cụ thể

Notebook này được thiết kế để hoạt động với dataset hình ảnh được tổ chức trong các thư mục con, với mỗi thư mục con đại diện cho một lớp (class).

Ví dụ, nếu `data_path` trỏ đến `../input/training-data`, thì cấu trúc dữ liệu có thể như sau:
```
training-data/
├── class_A/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_B/
│   ├── image_101.jpg
│   ├── image_102.jpg
│   └── ...
└── ...
```

Mỗi thư mục con (`class_A`, `class_B`, v.v.) sẽ được tự động nhận diện là một lớp bởi `tf.keras.utils.image_dataset_from_directory`.

### 4.2. Kỹ thuật Data Augmentation

Trong hàm `data_augmentation_layer()`, các kỹ thuật tăng cường dữ liệu sau được sử dụng:

* **`RandomFlip("horizontal")`**: Ngẫu nhiên lật hình ảnh theo chiều ngang. Điều này giúp mô hình trở nên bất biến với việc lật hình ảnh, một biến thể phổ biến trong dữ liệu thực tế.
* **`RandomRotation(0.1)`**: Ngẫu nhiên xoay hình ảnh trong một phạm vi nhất định (ví dụ: tối đa 10% của 360 độ). Điều này giúp mô hình học cách nhận diện các đối tượng ở các góc độ khác nhau.

Các kỹ thuật tăng cường dữ liệu này giúp mô hình học được các đặc trưng tổng quát hơn, giảm thiểu hiện tượng overfitting và cải thiện khả năng tổng quát hóa trên dữ liệu mới.

### 4.3. Kỹ thuật Preprocessing

Kỹ thuật tiền xử lý chính được sử dụng là:

* **`Rescaling(1./255)`**: Chuẩn hóa giá trị pixel. Các giá trị pixel gốc thường nằm trong khoảng `[0, 255]`. Lớp này chia tất cả các giá trị pixel cho 255, đưa chúng về khoảng `[0, 1]`.

**Lý do cho Preprocessing**:

* **Ổn định hóa quá trình huấn luyện**: Nhiều thuật toán tối ưu hóa (như gradient descent) hoạt động hiệu quả hơn khi các giá trị đầu vào nằm trong một phạm vi nhỏ và nhất quán.
* **Tăng tốc độ hội tụ**: Chuẩn hóa giúp hàm mất mát trở nên "mượt mà" hơn, giúp quá trình tối ưu hóa tìm được điểm tối ưu nhanh hơn.

Tóm lại, notebook này cung cấp một ví dụ toàn diện về việc xây dựng và huấn luyện một mô hình CNN, từ việc chuẩn bị dữ liệu với các kỹ thuật tăng cường và tiền xử lý, đến việc xây dựng kiến trúc mô hình, huấn luyện và đánh giá.