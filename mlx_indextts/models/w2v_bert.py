"""MLX inference for the released W2V-BERT 2.0 relative-key Conformer.

Architecture follows Hugging Face Transformers (Apache-2.0) and Meta's
facebook/w2v-bert-2.0 (MIT). IndexTTS consumes hidden_states[17], so inference
stops after block 17; unused layers are neither retained nor evaluated.
"""
import json
from pathlib import Path
import mlx.core as mx


class W2VBert:
    def __init__(self, directory, output_layer=17):
        directory = Path(directory)
        self.config = json.loads((directory/'config.json').read_text())
        c=self.config
        if c['position_embeddings_type']!='relative_key' or c.get('add_adapter') or c['hidden_act']!='swish':
            raise ValueError('Only the released relative-key, no-adapter W2V-BERT configuration is supported')
        if not 0 < output_layer <= c['num_hidden_layers']:
            raise ValueError('Invalid hidden-state layer')
        self.output_layer=output_layer
        self.weights={}
        for key,w in mx.load(str(directory/'model.safetensors')).items():
            if key.startswith('encoder.layers.') and int(key.split('.')[2]) >= output_layer:
                continue
            if key.endswith('.weight') and w.ndim==3:
                w=w.transpose(0,2,1)
            self.weights[key]=w
        mx.eval(self.weights)

    def linear(self,x,key):
        return x @ self.weights[key+'.weight'].T + self.weights[key+'.bias']

    def norm(self,x,key):
        return mx.fast.layer_norm(x,self.weights[key+'.weight'],self.weights[key+'.bias'],self.config['layer_norm_eps'])

    def ffn(self,x,key):
        x=self.linear(x,key+'.intermediate_dense')
        return self.linear(x*mx.sigmoid(x),key+'.output_dense')

    def attention(self,x,key,mask=None):
        b,t,d=x.shape;h=self.config['num_attention_heads'];hd=d//h
        q,k,v=[self.linear(x,key+'.linear_'+n).reshape(b,t,h,hd).transpose(0,2,1,3) for n in ('q','k','v')]
        c=self.config
        offsets=mx.clip(mx.arange(t)[None,:]-mx.arange(t)[:,None],-c['left_max_position_embeddings'],c['right_max_position_embeddings'])+c['left_max_position_embeddings']
        # Compute all 73 relative logits then gather by relative offset. Avoid
        # materializing a [time,time,head_dim] distance embedding per layer.
        relative=q @ self.weights[key+'.distance_embedding.weight'].T
        relative=mx.take_along_axis(relative,offsets[None,None,:,:],axis=-1)
        scores=(q @ k.transpose(0,1,3,2)+relative)*(hd**-0.5)
        if mask is not None:
            scores=mx.where(mask[:,None,None,:],scores,mx.finfo(mx.float32).min)
        value=mx.softmax(scores,axis=-1) @ v
        return self.linear(value.transpose(0,2,1,3).reshape(b,t,d),key+'.linear_out')

    def convolution(self,x,key,mask=None):
        x=self.norm(x,key+'.layer_norm')
        if mask is not None:x=mx.where(mask[:,:,None],x,0)
        x=mx.conv1d(x,self.weights[key+'.pointwise_conv1.weight'])
        a,b=mx.split(x,2,axis=-1);x=a*mx.sigmoid(b)
        w=self.weights[key+'.depthwise_conv.weight']
        x=mx.pad(x,[(0,0),(w.shape[1]-1,0),(0,0)])
        x=mx.conv1d(x,w,groups=w.shape[0])
        x=self.norm(x,key+'.depthwise_layer_norm');x=x*mx.sigmoid(x)
        return mx.conv1d(x,self.weights[key+'.pointwise_conv2.weight'])

    def __call__(self,features,mask=None,return_stages=False):
        x=self.linear(self.norm(features,'feature_projection.layer_norm'),'feature_projection.projection')
        if mask is not None:x=mx.where(mask[:,:,None],x,0)
        stages=[x] if return_stages else None
        for i in range(self.output_layer):
            key=f'encoder.layers.{i}'
            x=x+0.5*self.ffn(self.norm(x,key+'.ffn1_layer_norm'),key+'.ffn1')
            x=x+self.attention(self.norm(x,key+'.self_attn_layer_norm'),key+'.self_attn',mask)
            x=x+self.convolution(x,key+'.conv_module',mask)
            x=self.norm(x+0.5*self.ffn(self.norm(x,key+'.ffn2_layer_norm'),key+'.ffn2'),key+'.final_layer_norm')
            mx.eval(x) # bound graph and transient memory per block
            if return_stages:stages.append(x)
        return (x,stages) if return_stages else x
