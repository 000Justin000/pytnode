run_rnn_point_process:
	taskset --cpu-list 0 python rnn_point_processes.py --dataset poisson            --niters 10000 --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 1 python rnn_point_processes.py --dataset exponential_hawkes --niters 10000 --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 2 python rnn_point_processes.py --dataset self_inhibiting    --niters 10000 --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 3 python rnn_point_processes.py --dataset powerlaw_hawkes    --niters 10000 --batch_size 30 --nsave 100 --seed0 &

run_point_process:
	taskset --cpu-list 0 python point_processes.py --dataset poisson            --niters 5000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 1 python point_processes.py --dataset exponential_hawkes --niters 5000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 2 python point_processes.py --dataset self_inhibiting    --niters 5000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 3 python point_processes.py --dataset powerlaw_hawkes    --niters 5000 --jump_type read --batch_size 30 --nsave 100 --seed0 &

run_twitter:
	taskset --cpu-list 0 python tweet.py --dataset politics  --niters 1000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 1 python tweet.py --dataset movie     --niters 1000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 2 python tweet.py --dataset fight     --niters 1000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 3 python tweet.py --dataset bollywood --niters 1000 --jump_type read --batch_size 30 --nsave 100 --seed0 &

run_book_order:
	taskset --cpu-list 0 python book_order.py --niters 2500 --fold 0 --jump_type read --batch_size 10 --nsave 100 --seed0 &
	taskset --cpu-list 1 python book_order.py --niters 2500 --fold 1 --jump_type read --batch_size 10 --nsave 100 --seed0 &
	taskset --cpu-list 2 python book_order.py --niters 2500 --fold 2 --jump_type read --batch_size 10 --nsave 100 --seed0 &
	taskset --cpu-list 3 python book_order.py --niters 2500 --fold 3 --jump_type read --batch_size 10 --nsave 100 --seed0 &
	taskset --cpu-list 4 python book_order.py --niters 2500 --fold 4 --jump_type read --batch_size 10 --nsave 100 --seed0 &

run_stack_overflow:
	taskset --cpu-list 0 python stack_overflow.py --niters 2500 --fold 0 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 1 python stack_overflow.py --niters 2500 --fold 1 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 2 python stack_overflow.py --niters 2500 --fold 2 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 3 python stack_overflow.py --niters 2500 --fold 3 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 4 python stack_overflow.py --niters 2500 --fold 4 --jump_type read --batch_size 30 --nsave 500 --seed0 &

run_mimic2:
	taskset --cpu-list 0 python mimic2.py --niters 10000 --fold 0 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 1 python mimic2.py --niters 10000 --fold 1 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 2 python mimic2.py --niters 10000 --fold 2 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 3 python mimic2.py --niters 10000 --fold 3 --jump_type read --batch_size 30 --nsave 500 --seed0 &
	taskset --cpu-list 4 python mimic2.py --niters 10000 --fold 4 --jump_type read --batch_size 30 --nsave 500 --seed0 &

run_bitcoin_trust:
	taskset --cpu-list 0 python bitcoin_trust.py --dataset alpha  --niters 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
	taskset --cpu-list 1 python bitcoin_trust.py --dataset otc    --niters 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &

run_stocks_news:
	taskset --cpu-list 1 python stocks_news.py --niters 10000 --jump_type read --batch_size 30 --nsave 50 --seed0 &

run_stocks_news0:
	taskset --cpu-list 3 python stocks_news0.py --niters 10000 --jump_type read --batch_size 30 --nsave 50 --seed0 &

run_earthquake:
	taskset --cpu-list 1 python earthquake.py --niter 10000 --jump_type read --batch_size 8 --nsave 50 --seed0 &

run_point_process_gaussian:
	taskset --cpu-list 1 python point_processes_gaussian.py --datase exponential_hawkes --niter 3000 --jump_type read --batch_size 30 --nsave 100 --seed0 &
