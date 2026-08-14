'''
by lyuwenyu
'''
import time 
import json
import datetime
import os

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    @staticmethod
    def _save_checkpoint_atomic(state, path):
        """Write one recoverable checkpoint without accumulating epoch files."""
        if not dist.is_main_process():
            return
        temporary_path = path.with_name(f'.{path.name}.tmp')
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }
        best_stat_path = self.output_dir / 'best_stat.json'
        if args.yaml_cfg.get('checkpoint_mode') == 'best_last' and best_stat_path.exists():
            with best_stat_path.open('r') as f:
                best_stat = json.load(f)

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler,
                accumulation_steps=args.yaml_cfg.get('accumulation_steps', 1))

            self.lr_scheduler.step()
            
            checkpoint_state = None
            if self.output_dir and args.yaml_cfg.get('checkpoint_mode') == 'best_last':
                checkpoint_state = self.state_dict(epoch)
                self._save_checkpoint_atomic(
                    checkpoint_state, self.output_dir / 'last.pth')
            elif self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            best_metric = args.yaml_cfg.get('best_metric', 'coco_eval_bbox')
            if best_metric not in test_stats:
                raise KeyError(
                    f'best metric {best_metric!r} not found in {tuple(test_stats)}')
            previous_best = float(best_stat.get(best_metric, float('-inf')))
            current_metric = float(test_stats[best_metric][0])
            is_best = current_metric > previous_best

            # TODO 
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            if self.output_dir and args.yaml_cfg.get('checkpoint_mode') == 'best_last':
                if is_best:
                    self._save_checkpoint_atomic(
                        checkpoint_state, self.output_dir / 'best.pth')
                if dist.is_main_process():
                    temporary_stat = best_stat_path.with_suffix('.json.tmp')
                    with temporary_stat.open('w') as f:
                        json.dump(best_stat, f, indent=2)
                        f.write('\n')
                    os.replace(temporary_stat, best_stat_path)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
