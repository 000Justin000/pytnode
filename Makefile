run_rnn_point_process:
	taskset --cpu-list 0 python rnn_point_processes.py --dataset poisson            --niters 10000 --batch_size 30 --nsave 50 --seed0 &
	taskset --cpu-list 1 python rnn_point_processes.py --dataset exponential_hawkes --niters 10000 --batch_size 30 --nsave 50 --seed0 &
	taskset --cpu-list 2 python rnn_point_processes.py --dataset self_inhibiting    --niters 10000 --batch_size 30 --nsave 50 --seed0 &
	taskset --cpu-list 3 python rnn_point_processes.py --dataset powerlaw_hawkes    --niters 10000 --batch_size 30 --nsave 50 --seed0 &

run_point_process:
	taskset --cpu-list 0 python point_processes.py --dataset poisson            --niters 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 1 python point_processes.py --dataset exponential_hawkes --niters 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 2 python point_processes.py --dataset self_inhibiting    --niters 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 3 python point_processes.py --dataset powerlaw_hawkes    --niters 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &

run_book_order:
	taskset --cpu-list 0 python book_order.py --niters 3000 --jump_type read --batch_size 20 --nsave 100 --seed0 &

run_coupled_oscillators:
	taskset --cpu-list 0 python coupled_oscillators.py --niters 60000 --batch_size 300 --nsave 100 --seed0 &
