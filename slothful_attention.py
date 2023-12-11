
import math
from torch import nn

REDUCTION_MODES = ["1D_AVG", "1D_MAX", "2D_AVG", "2D_MAX"]

def clip(num, min_value, max_value):
  return max(min(num, max_value), min_value)

class SlothfulAttention:
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "model": ("MODEL",),
          "time_decay": ("FLOAT", {"default": 1.0, "min": 0, "max": 4.0, "step": 0.1}),
          "keep_middle": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

          "in_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "in_depth_decay": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 4.0, "step": 0.1}),
          "in_strength": ("FLOAT", {"default": 6.0, "min": 0, "max": 50.0, "step": 0.5}),
          "in_k_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "in_v_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),

          "out_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "out_depth_decay": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 4.0, "step": 0.1}),
          "out_strength": ("FLOAT", {"default": 4.0, "min": 0, "max": 50.0, "step": 0.5}),
          "out_k_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "out_v_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
        }
      }
  RETURN_TYPES = ("MODEL",)
  FUNCTION = "patch_model"

  CATEGORY = "SlothfulAttention"

  def patch_model(self, model, time_decay, keep_middle,
                  in_mode, in_depth_decay, in_strength, in_k_blend, in_v_blend,
                  out_mode, out_depth_decay, out_strength, out_k_blend, out_v_blend):

    # strength, depth_decay, time_decay から poolingのstrideを計算する
    def get_stride(q, extra_options):

      is_output = extra_options['block'][0] == 'output'

      depth_decay = out_depth_decay if is_output else in_depth_decay
      strength = out_strength if is_output else in_strength

      is_xl = extra_options['n_heads'] != 8

      # FIXME: timestep -> sigma の計算式を調べて正確にする
      # 2.68202423895704 to -2.0065569962006498 on 40 steps
      # とりあえず 2.8 to -2.2 の範囲として計算する
      time = (math.log(extra_options['sigmas'][0].item()) + 2.2) / 5
      time = max([0, min([1, time])])

      original_samples = extra_options['original_shape'][2] * extra_options['original_shape'][3]

      if depth_decay < 0:
        depth_decay *= -1

        # sdxlは 1:4が最も深い、sd1.5は 1:8が最も深い
        sample_rate = original_samples / q.shape[1] / 16 if is_xl else original_samples / q.shape[1] / 64

        # sample_rateは 寸法比の二乗。sqrt のかわりに depth_decay を 1/2にする 
        ratio = (sample_rate ** (depth_decay / 2)) * (time ** time_decay)
      else:
        # sdxlは一番上の層にattentionがないので、２層目 (1/2) から計算する
        sample_rate = 4 * q.shape[1] / original_samples if is_xl else q.shape[1] / original_samples

        # sample_rateは 寸法比の二乗。sqrt のかわりに depth_decay を 1/2にする 
        ratio = (sample_rate ** (depth_decay / 2)) * (time ** time_decay)

      return strength * ratio

    def get_blend(extra_options):      
        is_output = extra_options['block'][0] == 'output'
  
        k_blend = out_k_blend if is_output else in_k_blend
        v_blend = out_v_blend if is_output else in_v_blend
  
        return (k_blend, v_blend)
    
    def attn_patch(q, k, v, extra_options):
      # attn1にpatchしているので、常に self-attentionのはず
      # if q.shape != v.shape: # maybe cross attention
      #   return q, k, v

      # print(extra_options)

      if keep_middle and extra_options['block'][0] == 'middle':
        return q, k, v

      # print(q.shape, extra_options['block'], math.log(extra_options['sigmas'][0].item()))
      stride = get_stride(q, extra_options)

      if stride <= 1: # 1以下のときはpoolingしない
        return q, k, v

      is_output = extra_options['block'][0] == 'output'
      mode = out_mode if is_output else in_mode

      k_blend, v_blend = get_blend(extra_options)
      # print(f"#{extra_options['block']}: {mode} {stride}, {size}")

      if mode == "1D_AVG" or mode == "1D_MAX":
        size = (math.ceil(stride), 1)
        one = nn.functional.avg_pool2d(k, (1, 1), stride=size)

        if k_blend == 1 and v_blend == 1:
          return q, one, one
        
        pool = nn.functional.avg_pool2d(k, size, stride=size) if mode == "1D_AVG" else nn.functional.max_pool2d(k, size, stride=size)

        # poolingの結果、サイズが変わってしまうので、paddingして合わせる
        one = nn.functional.pad(one, (0, 0, 0, pool.shape[1] - one.shape[1]))

        k = one * (1 - k_blend) + pool * k_blend
        v = one * (1 - v_blend) + pool * v_blend
        return q, k, v
      
      else: # 2D
        original_samples = extra_options['original_shape'][2] * extra_options['original_shape'][3]
        sample_ratio = math.sqrt(q.shape[1] / original_samples)
        w = round(extra_options['original_shape'][3] * sample_ratio)
        h = q.shape[1] // w

        sty = math.ceil(math.sqrt(stride))
        stx = math.ceil(stride / sty)
        size = (stx, sty)

        k2d = k.permute([0, 2, 1]).reshape([k.shape[0], k.shape[2], h, w])

        one2d = nn.functional.avg_pool2d(k2d, (1, 1), stride=size)

        if k_blend == 1 and v_blend == 1:
          one = one2d.flatten(2).permute([0, 2, 1])
          # print(f"SlothfulAttention: {extra_options['block']}, {mode}, {q.shape[1]} -> {one.shape[1]}")
          return q, one, one

        pool2d = nn.functional.avg_pool2d(k2d, size, stride=size) if mode == "2D_AVG" else nn.functional.max_pool2d(k2d, size, stride=size)

        # poolingの結果、サイズが変わってしまうので、one2dをpaddingして合わせる
        one2d = nn.functional.pad(one2d, (0, pool2d.shape[3] - one2d.shape[3], 0, pool2d.shape[2] - one2d.shape[2]))

        one = one2d.flatten(2).permute([0, 2, 1])
        pool = pool2d.flatten(2).permute([0, 2, 1])

        k = one * (1 - k_blend) + pool * k_blend
        v = one * (1 - v_blend) + pool * v_blend
        # print(f"SlothfulAttention: {extra_options['block']}, {mode}, {q.shape[1]} -> {one.shape[1]}")
        return q, k, v

    model_patched = model.clone()
    model_patched.set_model_attn1_patch(attn_patch)
    # model_patched.set_model_attn2_patch(attn_patch)

    return (model_patched,)

class SlothfulAttentionSimple(SlothfulAttention):
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "model": ("MODEL",),
          "time_decay": ("FLOAT", {"default": 1.0, "min": 0, "max": 4.0, "step": 0.1}),
          "depth_decay": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 4.0, "step": 0.1}),

          "in_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "in_strength": ("FLOAT", {"default": 6.0, "min": 0, "max": 50.0, "step": 0.5}),
          "in_blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
          "out_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "out_strength": ("FLOAT", {"default": 4.0, "min": 0, "max": 50.0, "step": 0.5}),
          "out_blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),

        }
      }
  RETURN_TYPES = ("MODEL",)
  FUNCTION = "patch_model_simple"

  CATEGORY = "SlothfulAttention"

  # base model の patch_model を呼び出すs
  # def patch_model(self, model, time_decay, keep_middle,
  #                 in_mode, in_depth_decay, in_strength, in_k_blend, in_v_blend,
  #                 out_mode, out_depth_decay, out_strength, out_k_blend, out_v_blend):

  def patch_model_simple(self, model, time_decay, depth_decay,
                         in_mode, in_strength, in_blend,
                         out_mode, out_strength, out_blend):

    return self.patch_model(model, time_decay, True,
                            in_mode, depth_decay, in_strength, in_blend, in_blend,
                            out_mode, depth_decay, out_strength, out_blend, out_blend)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
  "SlothfulAttention": SlothfulAttention,
  "SlothfulAttentionSimple": SlothfulAttentionSimple
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SlothfulAttention": "Slothful Attention",
    "SlothfulAttentionSimple": "Slothful Attention Simple"
}

