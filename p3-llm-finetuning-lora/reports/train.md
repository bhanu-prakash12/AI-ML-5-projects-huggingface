# Training report

- LoRA r=8 alpha=16 on ['q_proj', 'k_proj', 'v_proj', 'o_proj']; trainable 1,081,344 of 495,114,112 (0.22%)
- lr 0.0002, 1 epoch, batch 1 x accum 8, fp32, grad checkpointing; steps 37; wall 5.2 min (this run); peak RAM 3.43 GB
- Train loss: first-5 mean 2.073 -> last-5 mean 1.785 (falling: True)
- Val loss: before 2.1342029571533203 -> after 1.917

| step | eval_loss |
|---|---|
| 10 | 1.987 |
| 20 | 1.934 |
| 30 | 1.918 |
| 37 | 1.917 |

Train loss per step: 2.04, 1.87, 2.40, 1.96, 2.10, 1.84, 1.57, 1.80, 1.78, 1.76, 1.89, 1.96, 1.89, 1.88, 1.94, 1.79, 1.83, 1.94, 1.70, 1.85, 1.82, 1.58, 1.97, 1.72, 2.00, 1.82, 1.89, 1.99, 1.69, 1.77, 1.91, 1.61, 1.96, 1.73, 1.72, 1.71, 1.80
