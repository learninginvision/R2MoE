# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================
#
# CreativeML Open RAIL-M License
#
# Section I: PREAMBLE

# Multimodal generative models are being widely adopted and used, and have the potential to transform the way artists, among other individuals, conceive and benefit from AI or ML technologies as a tool for content creation.

# Notwithstanding the current and potential benefits that these artifacts can bring to society at large, there are also concerns about potential misuses of them, either due to their technical limitations or ethical considerations.

# In short, this license strives for both the open and responsible downstream use of the accompanying model. When it comes to the open character, we took inspiration from open source permissive licenses regarding the grant of IP rights. Referring to the downstream responsible use, we added use-based restrictions not permitting the use of the Model in very specific scenarios, in order for the licensor to be able to enforce the license in case potential misuses of the Model may occur. At the same time, we strive to promote open and responsible research on generative models for art and content generation.

# Even though downstream derivative versions of the model could be released under different licensing terms, the latter will always have to include - at minimum - the same use-based restrictions as the ones in the original license (this license). We believe in the intersection between open and responsible AI development; thus, this License aims to strike a balance between both in order to enable responsible open-science in the field of AI.

# This License governs the use of the model (and its derivatives) and is informed by the model card associated with the model.

# NOW THEREFORE, You and Licensor agree as follows:

# 1. Definitions

# - "License" means the terms and conditions for use, reproduction, and Distribution as defined in this document.
# - "Data" means a collection of information and/or content extracted from the dataset used with the Model, including to train, pretrain, or otherwise evaluate the Model. The Data is not licensed under this License.
# - "Output" means the results of operating a Model as embodied in informational content resulting therefrom.
# - "Model" means any accompanying machine-learning based assemblies (including checkpoints), consisting of learnt weights, parameters (including optimizer states), corresponding to the model architecture as embodied in the Complementary Material, that have been trained or tuned, in whole or in part on the Data, using the Complementary Material.
# - "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.
# - "Complementary Material" means the accompanying source code and scripts used to define, run, load, benchmark or evaluate the Model, and used to prepare data for training or evaluation, if any. This includes any accompanying documentation, tutorials, examples, etc, if any.
# - "Distribution" means any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access.
# - "Licensor" means the copyright owner or entity authorized by the copyright owner that is granting the License, including the persons or entities that may have rights in the Model and/or distributing the Model.
# - "You" (or "Your") means an individual or Legal Entity exercising permissions granted by this License and/or making use of the Model for whichever purpose and in any field of use, including usage of the Model in an end-use application - e.g. chatbot, translator, image generator.
# - "Third Parties" means individuals or legal entities that are not under common control with Licensor or You.
# - "Contribution" means any work of authorship, including the original version of the Model and any modifications or additions to that Model or Derivatives of the Model thereof, that is intentionally submitted to Licensor for inclusion in the Model by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Model, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
# - "Contributor" means Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Model.

# Section II: INTELLECTUAL PROPERTY RIGHTS

# Both copyright and patent grants apply to the Model, Derivatives of the Model and Complementary Material. The Model and Derivatives of the Model are subject to additional terms as described in Section III.

# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute the Complementary Material, the Model, and Derivatives of the Model.
# 3. Grant of Patent License. Subject to the terms and conditions of this License and where and as applicable, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this paragraph) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Model and the Complementary Material, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Model to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model and/or Complementary Material or a Contribution incorporated within the Model and/or Complementary Material constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for the Model and/or Work shall terminate as of the date such litigation is asserted or filed.

# Section III: CONDITIONS OF USAGE, DISTRIBUTION AND REDISTRIBUTION

# 4. Distribution and Redistribution. You may host for Third Party remote access purposes (e.g. software-as-a-service), reproduce and distribute copies of the Model or Derivatives of the Model thereof in any medium, with or without modifications, provided that You meet the following conditions:
# Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5. This provision does not apply to the use of Complementary Material.
# You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License;
# You must cause any modified files to carry prominent notices stating that You changed the files;
# You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model, Derivatives of the Model.
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions - respecting paragraph 4.a. - for use, reproduction, or Distribution of Your modifications, or for any such Derivatives of the Model as a whole, provided Your use, reproduction, and Distribution of the Model otherwise complies with the conditions stated in this License.
# 5. Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
# 6. The Output You Generate. Except as set forth herein, Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses. No use of the output can contravene any provision as stated in the License.

# Section IV: OTHER PROVISIONS

# 7. Updates and Runtime Restrictions. To the maximum extent permitted by law, Licensor reserves the right to restrict (remotely or otherwise) usage of the Model in violation of this License, update the Model through electronic means, or modify the Output of the Model based on updates. You shall undertake reasonable efforts to use the latest version of the Model.
# 8. Trademarks and related. Nothing in this License permits You to make use of Licensors’ trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between the parties; and any rights not expressly granted herein are reserved by the Licensors.
# 9. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Model and the Complementary Material (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Model, Derivatives of the Model, and the Complementary Material and assume any risks associated with Your exercise of permissions under this License.
# 10. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Model and the Complementary Material (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
# 11. Accepting Warranty or Additional Liability. While redistributing the Model, Derivatives of the Model and the Complementary Material thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
# 12. If any provision of this License is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.

# END OF TERMS AND CONDITIONS




# Attachment A

# Use Restrictions

# You agree not to use the Model or Derivatives of the Model:
# - In any way that violates any applicable national, federal, state, local or international law or regulation;
# - For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# - To generate or disseminate verifiably false information and/or content with the purpose of harming others;
# - To generate or disseminate personal identifiable information that can be used to harm an individual;
# - To defame, disparage or otherwise harass others;
# - For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
# - For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
# - To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# - For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
# - To provide medical advice and medical results interpretation;
# - To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).
from datetime import datetime
from networkx import selfloop_edges
import torch
from einops import rearrange, repeat
from torch import nn, einsum
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion
from ldm.util import default
from ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlock
from ldm.modules.attention import CrossAttention as CrossAttention
from ldm.util import log_txt_as_img, exists, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torchvision.utils import make_grid
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
import numpy as np
from lora_diffusion import *
from src.custom_modules import *
class CustomDiffusion(LatentDiffusion):
    def __init__(self,
                 freeze_model='crossattn-kv',
                 cond_stage_trainable=False,
                 add_token=False,num_tasks=5,loss_ortho=None,lamda=1,rank=4,lora_type="lora",
                 loss_text_ortho=None,
                 text_embedding_learning_rate=1e-4,lr_route=1e-4,use_concept_mask=False,TopK=3,thres1=0.9,lora_del_step=600,
                 *args, **kwargs):

        self.freeze_model = freeze_model
        self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        self.num_tasks=num_tasks
        self.task_id=1
        self.loss_ortho=loss_ortho
        self.rank=rank
        self.lamda=lamda
        self.lora_type=lora_type
        self.loss_text_ortho=loss_text_ortho
        self.model_LWF=None
        self.text_embedding_learning_rate=text_embedding_learning_rate
        self.lr_route=lr_route
        self.use_concept_mask=use_concept_mask
        self.TopK=TopK
        self.thres1=thres1
        self.task_prompt=None
        self.lora_del_step=lora_del_step
        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)
        
        if self.freeze_model == 'crossattn-lora-kv' or self.freeze_model == 'attn-lora' :
            for x in self.model.diffusion_model.named_parameters():
                x[1].requires_grad = False
                #print(x[0])
        elif self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif not ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0]):
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = True
                
        def change_checkpoint(model):
            for layer in model.children():
                if type(layer) == BasicTransformerBlock:
                    layer.checkpoint = False 
                else:
                    change_checkpoint(layer)

        change_checkpoint(self.model.diffusion_model)
        def new_forward(self, x, context=None, mask=None,task_id=-1,mask_value=None,gate_c=None):
            h = self.heads
            crossattn = False
            if context is not None:
                crossattn = True
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)
            if crossattn:
                modifier = torch.ones_like(k)
                modifier[:, :1, :] = modifier[:, :1, :]*0.
                k = modifier*k + (1-modifier)*k.detach()
                v = modifier*v + (1-modifier)*v.detach()
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            attn = sim.softmax(dim=-1)
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        
        def new_forward_lora(self, x, context=None, mask=None,task_id=-1,mask_value=None,gate_c=None):
            h = self.heads
            crossattn = False
            q = self.to_q(x)
            #########mask operation#########
            if context is not None and isinstance(gate_c, dict) and x.shape[1]>=256:
                k = self.to_k(context,gate_c["c_gate"], task_id=task_id)
                k1= self.to_k(context,gate_c["c_gate"], task_id=-1)
                v = self.to_v(context,gate_c["c_gate"], task_id=task_id)
                v1= self.to_v(context,gate_c["c_gate"], task_id=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), ( k1, v1))
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
                sim1=einsum('b i d, b j d -> b i j', q, k1) * self.scale
                attn = sim.softmax(dim=-1)
                attn1=sim1.softmax(dim=-1)
                #result=attn.view(attn.shape[0],int(attn.shape[1]**(1/2)),int(attn.shape[1]**(1/2)),attn.shape[-1])
                position = gate_c["position"][0]
                result = attn[:,:,position:position+2] 
                result=torch.sum(result,dim=(0,-1))
                median_value = torch.median(result)
                result = torch.where(result >= median_value, torch.tensor(1.0,device=result.device), torch.tensor(0.0,device=result.device))        
                resized_mask = result.unsqueeze(0).unsqueeze(2)
                out1 = einsum('b i j, b j d -> b i d', attn1, v1)
                out2= einsum('b i j, b j d -> b i d', attn, v)
                out=(1-resized_mask)*out1+resized_mask*out2
                import matplotlib.pyplot as plt 
                print(result.shape)
                plt.imshow(result.view(int(attn.shape[1]**(1/2)),int(attn.shape[1]**(1/2))).cpu(), cmap='gray')  
                # plt.colorbar()  
                # plt.title( 'Median Visualization')  
                # plt.axis('off')
                # plt.savefig('tensor_visualization.png', dpi=300, bbox_inches='tight', pad_inches=0) 
                plt.imsave(f"tensor_visualization.png", result.view(int(attn.shape[1]**(1/2)),int(attn.shape[1]**(1/2))).cpu(),cmap='gray')
                # plt.show()
                plt.clf()  
            elif context is not None and mask_value is not None and x.shape[1]>=256:
                full_shape=int(x.shape[1]**(1/2))
                out = 0
                resized_mask_tensor_list = []
                batch_size = context.shape[0]//2
                q = rearrange(q, 'b n (h d) -> (b h) n d', h=h) 
                for i in range(len(mask_value)):  
                    region_prompt = torch.cat((context[0].unsqueeze(0).repeat(batch_size,1,1),mask_value[i]['prompt'].unsqueeze(0).repeat(batch_size,1,1)), dim=0)
                    if i != len(mask_value)-1:
                        if mask_value[-1]['time_step']>900:
                            k = self.to_k(region_prompt,gate_c[i].unsqueeze(0), task_id=task_id,scale=1)
                            v = self.to_v(region_prompt,gate_c[i].unsqueeze(0), task_id=task_id,scale=1)
                            resized_mask_tensor = torch.nn.functional.max_pool2d(mask_value[i]['mask'].unsqueeze(0), kernel_size=int(64/full_shape), stride=int(64/full_shape))
                            
                        else:
                            k = self.to_k(region_prompt,gate_c[i].unsqueeze(0), task_id=task_id,scale=1)
                            v = self.to_v(region_prompt,gate_c[i].unsqueeze(0), task_id=task_id,scale=1)
                            resized_mask_tensor = torch.nn.functional.max_pool2d(mask_value[i]['mask'].unsqueeze(0), kernel_size=int(64/full_shape), stride=int(64/full_shape))  #mask_value[i]['mask']  (batch_size 64 64)
                            
                        resized_mask_tensor_list.append(resized_mask_tensor)
                    else:
                        k = self.to_k(region_prompt, task_id=-1)
                        v = self.to_v(region_prompt, task_id=-1)
                        resized_mask_tensor = 1-torch.stack(resized_mask_tensor_list,dim=0).sum(dim=0)
                    k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))
                    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
                    attn=sim.softmax(dim=-1)  #[batch_size*16, 4096, 77]
                    resized_mask_tensor_reshape = resized_mask_tensor.view(batch_size, full_shape*full_shape, 1).repeat(attn.shape[0]//batch_size, 1,1)                 
                    attn = attn * resized_mask_tensor_reshape  
                    out_mask=einsum('b i j, b j d -> b i d', attn, v)    
                    out += out_mask 
            else:
                if context is not None:
                    crossattn = True
                    if  gate_c is not None:
                        k = self.to_k(context,gate_c,task_id=task_id,scale=1)
                        v = self.to_v(context,gate_c,task_id=task_id,scale=1)
                        
                    else:
                        k = self.to_k(context,task_id=task_id)
                        v = self.to_v(context,task_id=task_id)
                else:
                        context = default(context, x)
                        k = self.to_k(context)
                        v = self.to_v(context)
                if crossattn:
                    modifier = torch.ones_like(k)
                    modifier[:, :1, :] = modifier[:, :1, :]*0.
                    k = modifier*k + (1-modifier)*k.detach()
                    v = modifier*v + (1-modifier)*v.detach()
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
                attn = sim.softmax(dim=-1) 
                out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        def change_forward(model):
            for layer in model.children():
                if type(layer) == CrossAttention:
                    if self.freeze_model == "crossattn-kv":
                        bound_method = new_forward.__get__(layer, layer.__class__)
                    else :
                        bound_method = new_forward_lora.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)
                else:
                    change_forward(layer)
        change_forward(self.model.diffusion_model)
        
    def configure_optimizers(self):
        lr = self.learning_rate
        lr_text_embedding=self.text_embedding_learning_rate
        lr_route=self.lr_route
        for x in self.model.diffusion_model.named_parameters():
           x[1].requires_grad = False
        params = []
        params_lora=[]
        params_route=[]
        print("train layers")
        if self.freeze_model == 'attn-lora':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'attn1' in x[0] or 'attn2' in x[0]:
                        if 'to_k' in x[0] or 'to_v' in x[0] or 'to_q' in x[0]:
                            if 'lora_down' in x[0] or 'lora_up' in x[0] or 'lora_route' in x[0]:
                                params += [x[1]]
                                x[1].requires_grad = True
                                print(x[0])
        elif self.freeze_model == 'crossattn-kv':
            
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]:
                        params_lora += [x[1]]
                        x[1].requires_grad = True
                        print(x[0])
        elif self.freeze_model == 'crossattn-lora-kv':
            if self.lora_type=="loramoeonegate" or self.lora_type=="continuallora"or self.lora_type=="loraEC" or self.lora_type=="R2MoE":
                for x in self.model.diffusion_model.named_parameters():
                    if 'transformer_blocks' in x[0]:
                        if 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]:
                            if 'lora_down' in x[0] or 'lora_up' in x[0]:
                                params_lora += [x[1]]
                                x[1].requires_grad = True
                                print("lora",x[0])
                            elif 'lora_route' in x[0]:
                                params_route += [x[1]]
                                x[1].requires_grad = True
                                print("route",x[0])
            elif self.lora_type=="lora":
                for x in self.model.diffusion_model.named_parameters():
                    if 'transformer_blocks' in x[0]:
                        if 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]:
                            if 'lora_down' in x[0] or 'lora_up' in x[0] or 'lora_omegas' in x[0]:
                                params += [x[1]]
                                x[1].requires_grad = True
                                print(x[0])       
            
                            
        else:
            params = list(self.model.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.add_token:
                #params = params + list(self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters())
                params_t =list(self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters())
            else:
                params = params + list(self.cond_stage_model.parameters())

        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        #opt = torch.optim.AdamW(params, lr=lr)
        # opt = torch.optim.AdamW( [{'params': params, 'lr': lr},
        #                        {'params': params_t, 'lr': lr_text_embedding}]
        #                        )
        opt = torch.optim.AdamW( [{'params': params_lora, 'lr': lr},
                                  {'params': params_route, 'lr': lr_route},
                                {'params': params_t, 'lr': lr_text_embedding}]
                                )
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
    def apply_model_LWF(self, x_noisy, t, cond,task_id=-1, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop mask_value from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model_LWF(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model_LWF(x_noisy, t,task_id=task_id,**cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon
    def get_zero_lora(self,model,task_id,input,device):
        output = 0.0
        for layer in model.children():
            if type(layer)==ContinualLoraInjectedLinear or type(layer)==ContinualLoraMoeOneGateInjectedLinear or type(layer)==ContinualLora or type(layer)==ContinualCLora or type(layer)==R2MoE:
                    output+=layer.del_lora(input=input,task_id=task_id)
            elif list(layer.children()):
                    output += self.get_zero_lora(layer,task_id,input,device)
        return output
    def reset_grad(self,model,device):
        output = 0.0
        for layer in model.children():
            if type(layer)==ContinualLoraInjectedLinear or type(layer)==ContinualLoraMoeOneGateInjectedLinear or type(layer)==ContinualLora or type(layer)==ContinualCLora or type(layer)==R2MoE:
                    layer.requires_grad=True
            elif list(layer.children()):
                    output = self.reset_grad(layer,device)
        return output
    def get_loss_text_ortho(self,model,task_id, loss_type,device):
        total_loss_text_ortho = 0.0
        for layer in model.children():
            if type(layer)==FrozenCLIPEmbedderWrapper:
                    total_loss_text_ortho+=layer.loss_ortho_text(task_id=task_id,loss_type=loss_type,device=device)
            elif list(layer.children()):
                    total_loss_text_ortho += self.get_loss_text_ortho(layer,task_id, loss_type,device)
        return total_loss_text_ortho
    
    def get_loss_ortho(self,model,task_id, loss_type,device):
        total_loss_ortho = 0.0
        for layer in model.children():
            if type(layer)==ContinualLoraInjectedLinear or type(layer)==ContinualLoraMoeOneGateInjectedLinear or type(layer)==ContinualLora or type(layer)==ContinualCLora or type(layer)==R2MoE:
                    total_loss_ortho+=layer.loss_ortho(task_id=task_id,loss_type=loss_type,device=device)
                    
            elif list(layer.children()):
                    total_loss_ortho += self.get_loss_ortho(layer,task_id, loss_type,device)
        return total_loss_ortho
    
    def get_loss_Hadamard(self,model,task_id, device):
        total_loss_Hadamard = 0.0
        for layer in model.children():
            if type(layer)==ContinualLoraInjectedLinear or type(layer)==ContinualLora:
                    total_loss_Hadamard+=layer.loss_Hadamard(task_id=task_id, device=device)
            elif list(layer.children()):
                    total_loss_Hadamard += self.get_loss_Hadamard(layer, task_id, device)
        return total_loss_Hadamard
        
            
                
    def p_losses(self, x_start, cond, t, mask=None,concept_mask=None, task_id=-1, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond,task_id=task_id)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False)
        #if self.loss_text_ortho=="FA" or self.loss_text_ortho=="FA1":
            #loss_text_ortho = 0*self.get_loss_text_ortho(self.cond_stage_model,task_id,self.loss_text_ortho,device=self.device)
        if  self.loss_ortho == "rdm":
            loss_ortho = self.lamda*self.get_loss_ortho(self.model.diffusion_model,task_id,self.loss_ortho,device=self.device) 
            print("loss_ortho",loss_ortho)
        elif self.loss_ortho == "LWF":
            if task_id>1:
                model_output_LWF = self.lamda*self.apply_model_LWF(x_noisy, t, cond,task_id=task_id)
                outputs_S = F.softmax(model_output/2,dim=1)
                outputs_T = F.softmax(model_output_LWF/2, dim=1)
                loss_ortho = outputs_T.mul(-1 * torch.log(outputs_S))
                loss_ortho = loss_ortho.sum(1)
                loss_ortho=loss_ortho.mean()*4
                print(loss_ortho)
            else:
                loss_ortho=0
        elif self.loss_ortho == "Hadamard":
            loss_ortho = self.lamda*self.get_loss_Hadamard(self.model.diffusion_model,task_id,device=self.device)
            print("Hadamard",loss_ortho)
        else:
            loss_ortho=0
        if self.use_concept_mask:
            loss_simple = (loss_simple*concept_mask).sum([1, 2, 3])/concept_mask.sum([1, 2, 3])
        elif mask is not None:
            loss_simple = (loss_simple*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_simple = loss_simple.mean([1, 2, 3])
            
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = (self.logvar.to(self.device))[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False)
        if self.use_concept_mask:
            loss_vlb = (loss_vlb*concept_mask).sum([1, 2, 3])/concept_mask.sum([1, 2, 3])
        elif mask is not None:
            loss_vlb = (loss_vlb*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_vlb = loss_vlb.mean([1, 2, 3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        #print("loss_ortho",loss_ortho)
        loss+=loss_ortho
        loss_dict.update({f'{prefix}/loss_ortho': loss_ortho})
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict

    @torch.no_grad()
    def get_input_withmask(self, batch, **args):
        out = super().get_input(batch, self.first_stage_key, **args)
        mask = batch["mask"]
        concept_mask = batch["concept_mask"]
        if len(concept_mask.shape) == 3:
            concept_mask=concept_mask[..., None]
        if len(mask.shape) == 3:
            mask = mask[..., None]
        concept_mask=rearrange(concept_mask, 'b h w c -> b c h w')
        concept_mask = concept_mask.to(memory_format=torch.contiguous_format).float()
        mask = rearrange(mask, 'b h w c -> b c h w')
        mask = mask.to(memory_format=torch.contiguous_format).float()
        out += [mask]
        out += [concept_mask]
        return out

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            if self.self.global_step+1%self.trainer.max_steps==0:
                self.optimizers().param_groups[0]['lr']=self.learning_rate
            count=int(self.num_tasks*self.global_step/self.max_steps)
            train_batch = batch[count]
            loss, loss_dict = self.shared_step(train_batch)
        else:
            train_batch = batch
            loss, loss_dict = self.shared_step(train_batch)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return loss

    def shared_step(self, batch, **kwargs):
        x, c, mask,concept_mask = self.get_input_withmask(batch, **kwargs)
        print(c)
        task_id = batch["task_id"][0]
        #print(batch["task_id"])
        loss = self(x, c, mask=mask,concept_mask=concept_mask, task_id=task_id)
        return loss
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        print(f"========== learning rate {self.optimizers().param_groups[0]['lr']}==========")
        print(f"========== Training on task {self.task_id} ==========")
    
    def on_train_epoch_end(self):
        if self.lora_del_step is not None:
            if self.global_step==self.lora_del_step*self.trainer.max_steps:
                print(f"========== Task {self.task_id} Lora Del ==========")
                input1=self.get_learned_conditioning(self.task_prompt)
                a=self.get_zero_lora(self.model.diffusion_model,self.task_id,input1,device=self.device)
                print(a)
            if self.global_step==self.trainer.max_steps:
                print(f"========== Task {self.task_id} reset_grad ==========")
                a=self.reset_grad(self.model.diffusion_model,device=self.device)
    
        if self.global_step ==self.trainer.max_steps:
            print(f"========== Training on task {self.task_id} finished ==========")
            
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                unconditional_guidance_scale=6.
                unconditional_conditioning = self.get_learned_conditioning(len(c) * [""])
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                        unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples_scaled"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
