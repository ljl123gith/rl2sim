import torch
from legged_gym import LEGGED_GYM_ROOT_DIR

# load the trained policy jit model
policy_jit_path = f'{LEGGED_GYM_ROOT_DIR}/logs/v1_trot/exported/policies/policy_15000.pt'
policy_jit_model = torch.jit.load(policy_jit_path)

#set the model to evalution mode
policy_jit_model.eval()

# creat a fake input to the model
test_input_tensor = torch.randn(1,141)  

#specify the path and name of the output onnx model
policy_onnx_model = f'{LEGGED_GYM_ROOT_DIR}/sim2real_deploy/v1_trot_policy.onnx'

#export the onnx model
torch.onnx.export(policy_jit_model,               
                  test_input_tensor,       
                  policy_onnx_model,   # params below can be ignored
                  export_params=True,   
                  opset_version=11,     
                  do_constant_folding=True,  
                  input_names=['input'],    
                  output_names=['output'],  
                  )