# LoRA学習例（あみたろデータセット）

この例では、[あみたろの声素材工房様](https://amitaro.net/)で公開されている音声データを使用して、T5Gemma-TTSモデルのLoRA微調整を行う手順を示します。

## データセットの準備

あみたろの声素材工房様から各種音声データをダウンロードし、前処理をします。

1. データを一通りダウンロードし解凍

    - [あみたろの声素材（セリフ素材）一括ダウンロード](https://amitaro.net/voice/voice_dl/serifzip/)ページより、声素材（執筆時点では`amitarovoice_20251119_01.zip`）を一括ダウンロードします。利用前に声素材の[利用規約](https://amitaro.net/voice/voice_rule/)をご確認ください。
    - [あみたろのあみたろコーパス読み上げ音声](https://amitaro.net/voice/corpus-list/amitaro/)ページより、あみたろコーパス読み上げ音声（執筆時点では`amitarocorpus_amitaro_1.0.zip`）をダウンロードします。利用前に同ページの利用規約をご確認ください。
    - [あみたろのITAコーパス読み上げ音声](https://amitaro.net/voice/corpus-list/ita/)ページより、ITAコーパス読み上げ音声6種類（執筆時点では`ITAcorpus_amitaro_2.1.zip`、`ITAcorpus_amitaro_runrun.zip`、`ITAcorpus_amitaro_yofukashi_1.1.zip`、`ITAcorpus_amitaro_punsuka_1.0.zip`、`ITAcorpus_amitaro_sasayaki_A1.0.zip`、`ITAcorpus_amitaro_sasayaki_B1.0.zip`）をダウンロードします。利用前に同ページの利用規約をご確認ください。
    - [あみたろのMANAコーパス読み上げ音声](https://amitaro.net/voice/corpus-list/mana-corpus/)ページより、MANAコーパス読み上げ音声5種類（執筆時点では`MANAcorpus_amitaro_1.1.zip`、`MANAcorpus_amitaro_runrun.zip`、`MANAcorpus_amitaro_yofukashi.zip`、`MANAcorpus_amitaro_sasayaki_A.zip`、`MANAcorpus_amitaro_sasayaki_B.zip`）をダウンロードします。利用前に同ページの利用規約をご確認ください。
    - ダウンロードしたzipファイルをすべて解凍します。
        ```bash
        for f in *.zip; do out="${f%.zip}"; echo "[INFO] $f -> $out"; rm -rf "$out"; unzip -O cp932 "$f" -d "$out"; done
        ```
    - [ITAコーパスのリポジトリ](https://github.com/mmorise/ita-corpus)から、transcriptionのデータ（`emotion_transcript_utf8.txt`、`recitation_transcript_utf8.txt`）をダウンロードします。
    - [MANAコーパスのリポジトリ](https://github.com/shirowanisan/coeiroink-corpus-manager)から、transcriptionのデータ（`mana-corpus.txt`）をダウンロードします。
    - これらを同じディレクトリに以下のように配置してください。
        ```text
        examples/amitaro/
        ├── ITAcorpus_amitaro_2.1
        ├── ITAcorpus_amitaro_punsuka_1.0
        ├── ITAcorpus_amitaro_runrun
        ├── ITAcorpus_amitaro_sasayaki_A1.0
        ├── ITAcorpus_amitaro_sasayaki_B1.0
        ├── ITAcorpus_amitaro_yofukashi_1.1
        ├── MANAcorpus_amitaro_1.1
        ├── MANAcorpus_amitaro_runrun
        ├── MANAcorpus_amitaro_sasayaki_A
        ├── MANAcorpus_amitaro_sasayaki_B
        ├── MANAcorpus_amitaro_yofukashi
        ├── amitarocorpus_amitaro_1.0
        ├── amitarovoice_20251119_01
        ├── gen_samples
        ├── emotion_transcript_utf8.txt
        ├── recitation_transcript_utf8.txt
        ├── mana-corpus.txt
        ├── prepare_amitaro_dataset.py
        ├── t5gemma_2b-2b-ft-lora-amitaro.sh
        └── README.md
            (zipファイルは省略)
        ```

2. データセットの前処理

    `prepare_amitaro_dataset.py`を使ってデータセットの前処理を行います。このスクリプトでは、いくつかの処理が行われます。
    - 各コーパスのテキストと音声ファイルの対応付け、テキストの正規化、XCodec2による音声のencode
    - ITA / MANAコーパスでは、スタイルごとにprefixテキストを付与（`normal:`、`runrun:`、`yofukashi:`など）
    - 声素材とあみたろコーパスには一律`normal:`のprefixを付与
    - 各コーパス内・スタイルがある場合はスタイル内でneighbor情報を生成

    ```bash
    python examples/amitaro/prepare_amitaro_dataset.py \
        --output-dir datasets/amitaro_dataset \
        --valid-ratio 0.01
    ```

    これにより、以下のようなデータセットが生成されます。

    ```text
    datasets/amitaro_dataset/
    ├── manifest_final
    │   ├── train.txt # 学習用のデータ一覧（約3.6時間分）
    │   └── valid.txt # 検証用のデータ一覧
    ├── text # 各データの読み上げテキストのデータ
    ├── xcodec2_1cb # 各データの音声をXCodec2でencodeしたtokenのデータ
    └── neighbors # 各データのneighbor情報
    ```

## 学習

上記で準備したデータセットを用いて実際にLoRAでのファインチューニングを行います。

まず、Hugging Faceより学習済みcheckpointをダウンロードします。

```bash
# 認証情報を設定
hf auth login # または、HF_TOKENの環境変数を設定
# checkpointをダウンロード
hf download Aratako/T5Gemma-TTS-2b-2b ckpt/pretrained.pth --local-dir ./
```

次に、以下のコマンドで学習を開始します。

```bash
# 事前にスクリプト中のwandb_entityを自分のものに変更してください
NUM_GPUS=1 bash examples/amitaro/t5gemma_2b-2b-ft-lora-amitaro.sh
```

GPUが複数ある場合は、`NUM_GPUS`の値を変更してください。また、OOMが発生する場合は`--gradient_accumulation_steps`の値を増やして調整してください。

参考までに、私の環境で学習を実行した際のWandBのログは[こちら](https://api.wandb.ai/links/aratako-lm/rmfkjxqf)にあります。

## モデルの変換と推論

学習が完了したLoRAモデルをHF形式に変換します。

```bash
python scripts/export_t5gemma_voice_hf_lora.py \
    --ckpt runs/amitaro/bundle.pth \
    --out t5gemma_voice_amitaro \
    --base_repo google/t5gemma-2b-2b-ul2 \
    --save_adapter_dir amitaro-adapter
```

変換後、このモデルを用いて推論を試してみます。

```bash
python inference_commandline_hf.py \
    --model_dir ./t5gemma_voice_amitaro \
    --target_text "normal:こんにちは、私はAIです。これは音声合成のテストです。" \
    --target_duration 5
```

別のスタイルでも試してみます。

```bash
python inference_commandline_hf.py \
    --model_dir ./t5gemma_voice_amitaro \
    --target_text "sasayaki_a:こんにちは、これは音声合成のテストです。" \
    --target_duration 5
```

私の環境で生成した音声サンプルの一例を以下に示します。

| スタイル | サンプル |
| :--- | :--- |
| normal | [sample_amitaro1.wav](./gen_samples/sample_amitaro1.wav) |
| sasayaki_a | [sample_amitaro2.wav](./gen_samples/sample_amitaro2.wav) |
