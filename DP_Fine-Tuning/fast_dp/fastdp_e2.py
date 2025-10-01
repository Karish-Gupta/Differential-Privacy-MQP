from fastdp import *

if __name__ == "__main__":
   # Model Configs
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   dataset_name = "squad"
   train_batch_size = 4
   eval_batch_size = 4
   # gradient_accumulation_steps = 8
   num_epochs = 5
   learning_rate = 2e-4
   max_input_length = 512
   max_target_length = 512
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   target_epsilon = 2.0
   train_size = 5000
   eval_size = 500


   fastdp = FastDPModel(
         model_name=model_name,
         dataset_name=dataset_name,
         train_batch_size=train_batch_size,
         eval_batch_size=eval_batch_size,
         # gradient_accumulation_steps=gradient_accumulation_steps,
         num_epochs=num_epochs,
         learning_rate=learning_rate,
         max_input_length=max_input_length,
         max_target_length=max_target_length,
         target_epsilon=target_epsilon,
         train_size=train_size
   )

   # Start GPU utilization logging using utils
   gpu_util_thread, gpu_util_stop_event, gpu_util_data = start_gpu_utilization_logging(interval=1.0)

   fastdp.preprocess_dataset(train_size=train_size, eval_size=eval_size, seed=101)
   fastdp.init_model()
   fastdp.train()
   
   print(f"Model: {model_name}")
   print(f"On device: {device}")
   print(f"Number of epochs: {num_epochs}")
   print(f"Train batch size: {train_batch_size}")
   print(f"Eval batch size: {eval_batch_size}")
   print(f"Learning rate: {learning_rate}")
   print(f"Max input length: {max_input_length}")
   print(f"Max target length: {max_target_length}")
   print(f"Traing size: {train_size}")
   print(f"Eval size: {eval_size}")
   print(f"Epsilon: {target_epsilon}")
   
   fastdp.evaluate()

   # Ouput GPU logging
   stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
   print_gpu_utilization_summary(gpu_util_data)