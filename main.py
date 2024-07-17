import re  # 正規表現操作を行うためのモジュール
import random  # 乱数生成のためのモジュール
import time  # 時間の計測や操作を行うためのモジュール
from statistics import mode  # 最頻値（モード）を計算するためのモジュール

from PIL import Image  # Python Imaging Library（PIL）を使って画像を処理するためのモジュール
import numpy as np  # 数値計算を効率的に行うためのライブラリ
import pandas as pd  # データ操作や解析を行うためのライブラリ
import torch  # PyTorch: ディープラーニングのためのフレームワーク
import torch.nn as nn  # ニューラルネットワークモジュール
import torchvision  # 画像処理用のPyTorchモジュール
from torchvision import transforms  # 画像変換用のモジュール

def set_seed(seed):
    random.seed(seed)  # randomモジュールのシードを設定して、乱数生成の再現性を確保します。
    np.random.seed(seed)  # NumPyモジュールのシードを設定して、乱数生成の再現性を確保します。
    torch.manual_seed(seed)  # PyTorchのCPU上での乱数生成のシードを設定して、再現性を確保します。
    torch.cuda.manual_seed(seed)  # PyTorchのGPU上での乱数生成のシードを設定して、再現性を確保します。
    torch.cuda.manual_seed_all(seed)  # 複数のGPUを使用する場合に、全てのGPUでの乱数生成のシードを設定します。
    torch.backends.cudnn.deterministic = True  # CuDNNを使用する場合の再現性を確保するために、決定論的な挙動を有効にします。
    torch.backends.cudnn.benchmark = False  # 再現性を確保するために、CuDNNのベンチマークモードを無効にします。

def process_text(text):
    text = text.lower()  # テキストを小文字に変換

    num_word_to_digit = {  # 数詞を数字に変換する辞書
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():  # 数詞を数字に置き換える
        text = text.replace(word, digit)

    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)  # 小数点のピリオドを削除

    text = re.sub(r'\b(a|an|the)\b', '', text)  # 冠詞を削除

    contractions = {  # 短縮形を正規の形に戻す辞書
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():  # 短縮形を変換
        text = text.replace(contraction, correct)

    text = re.sub(r"[^\w\s':]", ' ', text)  # 句読点をスペースに変換

    text = re.sub(r'\s+,', ',', text)  # 不要なスペースを削除

    text = re.sub(r'\s+', ' ', text).strip()  # 連続するスペースを1つに変換

    return text  # 処理したテキストを返す

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理を設定
        self.image_dir = image_dir  # 画像ファイルのディレクトリを設定
        self.df = pd.read_json(df_path)  # JSONファイルを読み込みDataFrameに変換
        self.answer = answer  # 回答が含まれるかどうかのフラグを設定

        self.question2idx = {}  # 質問文の単語をインデックスに変換する辞書
        self.answer2idx = {}  # 回答をインデックスに変換する辞書
        self.idx2question = {}  # インデックスを質問文の単語に変換する辞書
        self.idx2answer = {}  # インデックスを回答に変換する辞書

        for question in self.df["question"]:  # 質問文に含まれる単語を辞書に追加
            question = process_text(question)  # テキストを処理
            words = question.split(" ")  # 単語に分割
            for word in words:  # 単語を辞書に追加
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # インデックスを質問文に変換する辞書を作成

        if self.answer:  # 回答がある場合
            for answers in self.df["answers"]:  # 回答に含まれる単語を辞書に追加
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # インデックスを回答に変換する辞書を作成

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx  # 訓練データセットの質問文の辞書を更新
        self.answer2idx = dataset.answer2idx  # 訓練データセットの回答の辞書を更新
        self.idx2question = dataset.idx2question  # 訓練データセットの質問文の逆引き辞書を更新
        self.idx2answer = dataset.idx2answer  # 訓練データセットの回答の逆引き辞書を更新

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")  # 画像を読み込む
        image = self.transform(image)  # 画像を前処理する
        question = np.zeros(len(self.idx2question) + 1)  # 質問文のone-hotベクトルを初期化
        question_words = self.df["question"][idx].split(" ")  # 質問文を単語に分割
        for word in question_words:  # 質問文の単語をone-hotベクトルに変換
            try:
                question[self.question2idx[word]] = 1
            except KeyError:
                question[-1] = 1  # 未知語の場合

        if self.answer:  # 回答がある場合
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得
            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)  # 画像、質問、回答、最頻値の回答を返す
        else:
            return image, torch.Tensor(question)  # 画像と質問を返す

    def __len__(self):
        return len(self.df)  # データセットのサイズを返す

def VQA_criterion(batch_pred, batch_answers):
    total_acc = 0.  # 総合正解率を初期化
    for pred, answers in zip(batch_pred, batch_answers):  # バッチ内の各予測と対応する回答をループ
        acc = 0.  # 個々の正解率を初期化
        for i in range(len(answers)):  # 各回答についてループ
            num_match = 0  # 一致した回答の数を初期化
            for j in range(len(answers)):  # 他の回答についてループ
                if i == j:  # 自分自身との比較をスキップ
                    continue
                if pred == answers[j]:  # 予測と回答が一致した場合
                    num_match += 1  # 一致数をカウント
            acc += min(num_match / 3, 1)  # 正解率を計算し、最大値を1とする
        total_acc += acc / 10  # 正解率の平均を計算し、総合正解率に加算
    return total_acc / len(batch_pred)  # バッチ内の平均正解率を返す

class BasicBlock(nn.Module):
    expansion = 1  # 拡張係数を設定（BasicBlockの場合は1）

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 最初の畳み込み層
        self.bn1 = nn.BatchNorm2d(out_channels)  # バッチ正規化層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)  # 2番目の畳み込み層
        self.bn2 = nn.BatchNorm2d(out_channels)  # バッチ正規化層
        self.relu = nn.ReLU(inplace=True)  # 活性化関数

        self.shortcut = nn.Sequential()  # ショートカット（恒等写像）
        if stride != 1 or in_channels != out_channels:  # チャネル数が一致しない場合やストライドが1でない場合
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),  # 1x1の畳み込み層
                nn.BatchNorm2d(out_channels)  # バッチ正規化層
            )

    def forward(self, x):
        residual = x  # 入力を保持
        out = self.relu(self.bn1(self.conv1(x)))  # 畳み込み->バッチ正規化->ReLU
        out = self.bn2(self.conv2(out))  # 畳み込み->バッチ正規化
        out += self.shortcut(residual)  # ショートカットを加算
        out = self.relu(out)  # ReLUを適用
        return out

class BottleneckBlock(nn.Module):
    expansion = 4  # 拡張係数を設定（BottleneckBlockの場合は4）

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)  # 1x1の畳み込み層
        self.bn1 = nn.BatchNorm2d(out_channels)  # バッチ正規化層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 3x3の畳み込み層
        self.bn2 = nn.BatchNorm2d(out_channels)  # バッチ正規化層
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)  # 1x1の畳み込み層
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  # バッチ正規化層
        self.relu = nn.ReLU(inplace=True)  # 活性化関数

        self.shortcut = nn.Sequential()  # ショートカット（恒等写像）
        if stride != 1 or in_channels != out_channels * self.expansion:  # チャネル数が一致しない場合やストライドが1でない場合
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),  # 1x1の畳み込み層
                nn.BatchNorm2d(out_channels * self.expansion)  # バッチ正規化層
            )

    def forward(self, x):
        residual = x  # 入力を保持
        out = self.relu(self.bn1(self.conv1(x)))  # 1x1の畳み込み->バッチ正規化->ReLU
        out = self.relu(self.bn2(self.conv2(out)))  # 3x3の畳み込み->バッチ正規化->ReLU
        out = self.bn3(self.conv3(out))  # 1x1の畳み込み->バッチ正規化
        out += self.shortcut(residual)  # ショートカットを加算
        out = self.relu(out)  # ReLUを適用
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64  # 初期の入力チャネル数を設定
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 初めの畳み込み層
        self.bn1 = nn.BatchNorm2d(64)  # バッチ正規化層
        self.relu = nn.ReLU(inplace=True)  # 活性化関数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # プーリング層

        self.layer1 = self._make_layer(block, layers[0], 64)  # 最初のレイヤー
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)  # 2番目のレイヤー
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)  # 3番目のレイヤー
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)  # 4番目のレイヤー

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 平均プーリング層
        self.fc = nn.Linear(512 * block.expansion, 512)  # 全結合層

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]  # 最初のブロックを追加
        self.in_channels = out_channels * block.expansion  # チャネル数を更新
        for _ in range(1, blocks):  # 残りのブロックを追加
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)  # ブロックを順次結合してシーケンスを作成

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 畳み込み->バッチ正規化->ReLU
        x = self.maxpool(x)  # プーリング
        x = self.layer1(x)  # 最初のレイヤー
        x = self.layer2(x)  # 2番目のレイヤー
        x = self.layer3(x)  # 3番目のレイヤー
        x = self.layer4(x)  # 4番目のレイヤー
        x = self.avgpool(x)  # 平均プーリング
        x = x.view(x.size(0), -1)  # 平坦化
        x = self.fc(x)  # 全結合層
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18モデルを作成

def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])  # ResNet50モデルを作成

class VQAModel(nn.Module):
    def __init__(self, vocab_size, n_answer):
        super().__init__()
        self.resnet = ResNet18()  # ResNet18モデルを初期化
        self.text_encoder = nn.Linear(vocab_size, 512)  # 質問文をエンコードする全結合層

        self.fc = nn.Sequential(  # 最後の全結合層
            nn.Linear(1024, 512),  # 画像特徴量(512)と質問文特徴量(512)を結合したものを入力
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(inplace=True),  # 活性化関数ReLU
            nn.Dropout(0.4),  # Dropout
            nn.Linear(512, 256),  # 次の全結合層への変換
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(inplace=True),  # 活性化関数ReLU
            nn.Dropout(0.4),  # Dropout
            nn.Linear(256, n_answer)  # 最終的な出力層
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像特徴量を抽出（N, 512）
        question_feature = self.text_encoder(question)  # 質問文の特徴量を抽出（N, 512）
        x = torch.cat([image_feature, question_feature], dim=1)  # 画像特徴量と質問文特徴量を結合（N, 1024）
        x = self.fc(x)  # 全結合層を通す（N, n_answer）
        return x

def train(model, dataloader, optimizer, criterion, device):
    model.train()  # モデルを訓練モードに設定
    total_loss = 0  # 総損失を初期化
    total_acc = 0  # 総正解率を初期化
    simple_acc = 0  # シンプル正解率を初期化
    start = time.time()  # 訓練の開始時間を記録

    for image, question, answers, mode_answer in dataloader:  # データローダーからバッチを取得
        image, question, answers, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)  # デバイスにデータを転送
        pred = model(image, question)  # モデルにデータを入力して予測を取得
        loss = criterion(pred, mode_answer.squeeze())  # 損失を計算

        optimizer.zero_grad()  # 勾配をゼロにリセット
        loss.backward()  # 誤差逆伝播を実行
        optimizer.step()  # パラメータを更新

        total_loss += loss.item()  # 総損失を更新
        total_acc += VQA_criterion(pred.argmax(1), answers)  # 総正解率を更新
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # シンプル正解率を更新

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start  # 平均損失、正解率、シンプル正解率、訓練時間を返す

def eval(model, dataloader, optimizer, criterion, device):
    model.eval()  # モデルを評価モードに設定
    total_loss = 0  # 総損失を初期化
    total_acc = 0  # 総正解率を初期化
    simple_acc = 0  # シンプル正解率を初期化
    start = time.time()  # 評価の開始時間を記録

    with torch.no_grad():  # 評価中は勾配を計算しない
        for image, question, answers, mode_answer in dataloader:  # データローダーからバッチを取得
            image, question, answers, mode_answer = image.to(device), question.to(device), answers.to(device), mode_answer.to(device)  # デバイスにデータを転送
            pred = model(image, question)  # モデルにデータを入力して予測を取得
            loss = criterion(pred, mode_answer.squeeze())  # 損失を計算

            total_loss += loss.item()  # 総損失を更新
            total_acc += VQA_criterion(pred.argmax(1), answers)  # 総正解率を更新
            simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # シンプル正解率を更新

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start  # 平均損失、正解率、シンプル正解率、評価時間を返す

# データセットのパスやその他の設定
train_json_path = "./data/train.json"  # 訓練データのJSONファイルのパス
valid_json_path = "./data/valid.json"  # 検証データのJSONファイルのパス
train_image_dir = "./data/train"  # 訓練画像のディレクトリパス
valid_image_dir = "./data/valid"  # 検証画像のディレクトリパス
model_path = "model.pth"  # モデルパラメータを保存するファイルのパス
submission_path = "submission.npy"  # 提出用の予測結果を保存するファイルのパス
num_epoch = 20 # エポック数
batch_size = 128  # バッチサイズ
learning_rate = 0.001  # 学習率
weight_decay = 1e-5  # 重み減衰（L2正則化）
seed = 42  # ランダムシード

set_seed(seed)  # ランダムシードを設定
device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用するデバイスを設定

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像を224x224にリサイズ
    transforms.RandomHorizontalFlip(),  # 水平方向にランダムに反転
    transforms.RandomRotation(10),  # ランダムに回転 (±10度)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # カラージッター
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ノーマライゼーション
])

train_dataset = VQADataset(df_path=train_json_path, image_dir=train_image_dir, transform=transform)  # 訓練データセットを作成
test_dataset = VQADataset(df_path=valid_json_path, image_dir=valid_image_dir, transform=transform, answer=False)  # 検証データセットを作成
test_dataset.update_dict(train_dataset)  # 検証データセットの辞書を訓練データセットに合わせて更新

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 訓練データローダーを作成
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)  # 検証データローダーを作成

model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)  # モデルを初期化
criterion = nn.CrossEntropyLoss()  # 損失関数を設定
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # オプティマイザを設定

for epoch in range(num_epoch):  # エポック数だけループ
    train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)  # 訓練を実行し、損失、正解率、シンプル正解率、訓練時間を取得
    print(f"【{epoch + 1}/{num_epoch}】\n"  # 現在のエポック数を表示
          f"train time: {train_time:.2f} [s]\n"  # 訓練時間を表示
          f"train loss: {train_loss:.4f}\n"  # 訓練損失を表示
          f"train acc: {train_acc:.4f}\n"  # 訓練正解率を表示
          f"train simple acc: {train_simple_acc:.4f}")  # シンプル正解率を表示

model.eval()  # モデルを評価モードに設定
submission = []  # 提出用の予測結果を保存するリストを初期化

for image, question in test_loader:  # テストデータローダーからバッチを取得してループ
    image, question = image.to(device), question.to(device)  # デバイスにデータを転送
    pred = model(image, question)  # モデルにデータを入力して予測を取得
    pred = pred.argmax(1).cpu().item()  # 予測をクラスラベルに変換し、CPUに転送して数値に変換
    submission.append(pred)  # 予測結果をリストに追加

submission = [train_dataset.idx2answer[id] for id in submission]  # 予測結果のIDを回答テキストに変換
submission = np.array(submission)  # 予測結果をNumPy配列に変換
torch.save(model.state_dict(), model_path)  # モデルのパラメータをファイルに保存
np.save(submission_path, submission)  # 予測結果をファイルに保存
