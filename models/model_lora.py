import os
import torch
import numpy as np
from torch import nn
from transformers import T5EncoderModel, Swinv2Model, T5Config, logging, ResNetModel, T5ForConditionalGeneration
from models.vqgan import VQModel
from modules.losses import FocalLoss
logging.set_verbosity_error()

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.result_dir = args.result_dir
        
        # self.vae = VQModel(ckpt_path=args.vae_ckpt_path).requires_grad_(False)
        # self.vae.eval()
        # 言語埋め込み
        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).requires_grad_(False) # device_map="auto"
        self.language_model.eval()

        if "resnet" in args.image_model_name: # 事前学習用に書き換えたのでおそらく動かない
            self.image_model = ResNetModel.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)
        elif "swinv2" in args.image_model_name:
            self.image_model = Swinv2Model.from_pretrained(args.image_model_name, use_mask_token=args.pretrain).requires_grad_(args.image_model_train)
            # self.num_patches = (self.image_model.config.image_size // self.image_model.config.patch_size) ** 2
            self.num_patches = 16 ** 2


        # 追加(t5 decorderを事前学習済みモデルに変更)
        # 事前学習済みのモデルと設定をロード
        self.transformer = T5ForConditionalGeneration.from_pretrained(args.transformer_model_name)
        # # pretrained_config = pretrained_model.config
        

        # # 設定を変更してエンコーダの層数を2に設定
        # new_config = T5Config.from_pretrained(args.transformer_model_name)
        # # print(new_config)
        # new_config.num_layers = args.transformer_num_layers
        # new_config.d_model = args.transformer_d_model  # d_modelの値を事前学習済みモデルと同じにする
        # new_config.vacab_size = 32128+args.loc_vocab_size+args.image_vocab_size
        # # # new_config.is_gated_act = False
        # # # new_config.dense_act = "relu"
        # # # new_config.feed_forward_proj = "relu"
        # # # new_config.decoder_start_token_id=0
        # # # new_config.max_length=args.max_target_length
        # # # print(new_config)

        # new_model = T5ForConditionalGeneration(new_config)

        # # # 事前学習済みモデルからデコーダの重みを新しいモデルにコピー
        # new_model.decoder.load_state_dict(pretrained_model.decoder.state_dict())

        # # # デコーダの重みを固定
        # for param in new_model.decoder.parameters():
        #     param.requires_grad = False
         
        # # # # 出力層のみを学習可能にする
        # # # for param in new_model.lm_head.parameters():
        # # #     param.requires_grad = True

        # self.transformer = new_model


        if args.ffn: 
            self.language_ffn = nn.Linear(self.language_model.config.d_model, self.transformer.config.d_model)
            self.image_ffn = nn.Linear(self.image_model.num_features, self.transformer.config.d_model)
            # self.image_ffn = nn.Linear(self.image_model.last_hidden_states.shape[1], self.transformer.config.d_model)

    def forward(self, images, src_texts, tgt_texts=None, return_loss=True, num_beams=1, num_return_sequences=1, do_sample=False, image_mask_ratio=0.0):
        with torch.no_grad():
            language_attention_mask = torch.ones(src_texts.shape[0], src_texts.shape[1], device=self.language_model.device)
            language_attention_mask[src_texts == 0] = 0
            language_embeddings = self.language_model(src_texts, attention_mask=language_attention_mask).last_hidden_state

        if image_mask_ratio > 0: # 画像パッチにマスクをかける
            bool_masked_pos = self.random_patch_masking(len(images), image_mask_ratio)
        else:
            bool_masked_pos = None
        # image_embeddings = self.image_model(pixel_values=images, bool_masked_pos=bool_masked_pos).last_hidden_state
        image_embeddings = self.image_model(pixel_values=images).last_hidden_state

        if self.args.ffn:
            with torch.no_grad():
                language_embeddings = self.language_ffn(language_embeddings)
                image_embeddings = self.image_ffn(image_embeddings)

        image_attention_mask = torch.ones(image_embeddings.shape[0], image_embeddings.shape[1], device=self.image_model.device)
        concat_attention_mask = torch.cat((image_attention_mask, language_attention_mask), dim=1)
        concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)

        if return_loss:
            if image_mask_ratio > 0:
                pred = self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts, attenntion_mask=concat_attention_mask, decoder_attention_mask=target_attention_mask).logits
                b, t, v = pred.shape
                pred = pred.view(b*t, v)
                tgt_texts = tgt_texts.view(-1)
                loss = self.loss_fct(pred, tgt_texts)
                loss = loss.view(b, t)
                if loss.shape[1] - bool_masked_pos.shape[1] > 0:
                    add_mask = torch.ones(loss.shape[0], loss.shape[1] - bool_masked_pos.shape[1], device=self.image_model.device)
                    bool_masked_pos = torch.cat((bool_masked_pos, add_mask), dim=1)
                loss = loss * bool_masked_pos
                return loss.mean()
            else:
                target_attention_mask = torch.ones(tgt_texts.shape[0], tgt_texts.shape[1], device=self.transformer.device)
                target_attention_mask[tgt_texts == 0] = 1
                if self.args.loss == 'CrossEntropy':
                    loss = self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts, attention_mask=concat_attention_mask, decoder_attention_mask=target_attention_mask).loss
                    generated = self.transformer.generate(inputs_embeds=concated_embeddings, num_beams=num_beams, num_return_sequences=num_return_sequences, do_sample=do_sample, max_length=self.args.max_target_length)
                    return loss,generated
                elif self.args.loss == 'FocalLoss':
                    loss_fct = FocalLoss()
                    logits = self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts, attention_mask=concat_attention_mask, decoder_attention_mask=target_attention_mask).logits
                    loss = loss_fct(logits.view(-1,logits.shape[2]), tgt_texts.view(-1))
                    generated = self.transformer.generate(inputs_embeds=concated_embeddings, num_beams=num_beams, num_return_sequences=num_return_sequences, do_sample=do_sample, max_length=self.args.max_target_length)
                    return loss,generated
        else:
            # pred = self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts).logits
            # generated = torch.argmax(pred, dim=2)
            generated = self.transformer.generate(inputs_embeds=concated_embeddings, num_beams=num_beams, num_return_sequences=num_return_sequences, do_sample=do_sample, max_length=self.args.max_target_length)
            return generated
    
    def random_patch_masking(self, batch_size, image_mask_ratio):
        len_keep = int(self.num_patches * image_mask_ratio)
        noise = torch.rand(batch_size, self.num_patches, device=self.image_model.device)

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([batch_size, self.num_patches], device=self.image_model.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask
    
    def image_to_z(self, images):
        z = self.vae.get_codebook_indices(images) # VAEで中間表現を得る
        z_text = z.cpu().numpy().astype(str) # 文字列に変換
        z_text = np.char.add(np.char.add('<img_', z_text), '>') # <extra_id_0>のようにする
        z_text = [''.join(b) for b in z_text]
        return z_text, z
    
    def z_to_image(self, z):
        x = self.vae.decode_code(z)
        return x

    def save(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = {'transformer': self.transformer.state_dict()}
        if self.args.image_model_train:
            checkpoints['image_model'] = self.image_model.state_dict()
        if self.args.ffn:
            checkpoints['language_ffn'] = self.language_ffn.state_dict()
            checkpoints['image_ffn'] = self.image_ffn.state_dict()
        torch.save(checkpoints, result_path)

    def load(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = torch.load(result_path)
        self.transformer.load_state_dict(checkpoints['transformer'])
        if self.args.image_model_train:
            self.image_model.load_state_dict(checkpoints['image_model'])
        if self.args.ffn:
            self.language_ffn.load_state_dict(checkpoints['language_ffn'])
            self.image_ffn.load_state_dict(checkpoints['image_ffn'])