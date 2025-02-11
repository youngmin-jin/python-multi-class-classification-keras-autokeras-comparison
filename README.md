# 29/02/12 
Jin) 
- **Ia model** (Autokeras model) achieved an accuracy of **0.84** (max_trais=20, epochs=30)
  <details>
  <summary>Results</summary>
    <kbd><image src="https://github.com/user-attachments/assets/57a5dce3-dc63-457a-8224-c84cabfeb6cc" width="400"></kbd> <br><br>
    <kbd><image src="https://github.com/user-attachments/assets/d3863315-dd98-477b-a397-72938fabe625" width="400"></kbd> <br><br>
    <kbd><image src="https://github.com/user-attachments/assets/aac94868-6459-4b8f-837e-31550778e472" width="300"></kbd> <br><br>
    <kbd><image src="https://github.com/user-attachments/assets/e5797f71-d9e2-401b-bd8e-929360688676" width="400"></kbd> <br><br>
  </details>

<br><br>

- **Io2 model** (based on the Autokeras model structure with hyperparametertuning) achieved an accuracy of **0.82** (epochs=20/ learning_rate, drop_connect_rate, and batch_size (32 fixed) applied)
  <details>
  <summary>Results</summary>
    <kbd><image src="https://github.com/user-attachments/assets/f5a5171b-b387-4605-a519-1907fabd32f3" width="400"></kbd> <br><br>
    <kbd><image src="https://github.com/user-attachments/assets/c07551e8-4f1e-4705-a2e2-0027709c1024" width="400"></kbd> <br><br>
  </details>

  <details>
  <summary>Other hyperparameters that have been tested so far</summary>
    - random_translation (height_factor)=['0.02, 0.05, 0.1'] <br>
    - random_rotation=['0.02, 0.05, 0.1']<br>
    - drop_connect_rate=[0, 0.2, 0.5]<br>
    - batch_sized (fixed)= 16 or 32 or 64
  </details>
