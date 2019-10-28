import collections
import logging
import numpy as np
import torch

from attack.group_generator import EquallySplitGrouping


def sim_rectification_vector(model, vid, tentative_directions, n, sigma, target_class, rank_transform, sub_num,
                             group_gen, untargeted):
    with torch.no_grad():
        grads = torch.zeros(len(group_gen), device='cuda')
        count_in = 0
        loss_total = 0
        logging.info('sampling....')
        batch_loss = []
        batch_noise = []
        batch_idx = []

        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma

            all_noise = torch.cat([noise_list, -noise_list], 0)

            perturbation_sample = group_gen.apply_group_change(tentative_directions, all_noise)
            adv_vid_rs += perturbation_sample
            del perturbation_sample

            top_val, top_idx, logits = model(adv_vid_rs)
            if untargeted:
                loss = -torch.max(logits, 1)[0]
            else:
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target_class, dtype=torch.long,
                                                                              device='cuda').repeat(sub_num),
                                                         reduction='none')
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)
        batch_noise = torch.cat(batch_noise, 0)
        batch_idx = torch.cat(batch_idx)
        batch_loss = torch.cat(batch_loss)

        # Apply rank-based loss transformation
        if rank_transform:
            good_idx = torch.sum(batch_idx == target_class, 1).byte()
            changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))
            loss_order = torch.zeros(changed_loss.size(0), device='cuda')
            sort_index = changed_loss.sort()[1]
            loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)
            available_number = torch.sum(good_idx).item()
            count_in += available_number
            unavailable_number = n - available_number
            unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                       loss_order)) / unavailable_number if unavailable_number else torch.tensor(
                0., device='cuda')
            rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)
            grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),
                               0)
        else:
            idxs = (batch_idx == target_class).nonzero()
            valid_idxs = idxs[:, 0]
            valid_loss = torch.index_select(batch_loss, 0, valid_idxs)

            loss_total += torch.mean(valid_loss).item()
            count_in += valid_loss.size(0)
            noise_select = torch.index_select(batch_noise, 0, valid_idxs)
            grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),
                               0)

        if count_in == 0:
            return None, None
        logging.info('count in: {}'.format(count_in))
        return loss_total / count_in, grads


# Input video should be torch.tensor and its shape should be [num_frames, c, w, h].
# The input should be normalized to [0, 1]
def targeted_video_attack(vid_model, vid, target_vid, directions_generator, target_class,
                          rank_transform=False, starting_eps=1., eps=0.05,
                          delta_eps=0.5, max_lr=1e-2, min_lr=1e-3, sample_per_draw=50,
                          max_iter=10000, sigma=1e-5, sub_num_sample=12,
                          image_split=1):
    num_iter = 0
    adv_vid = target_vid.clone()
    cur_eps = starting_eps

    explore_succ = collections.deque(maxlen=5)
    reduce_eps_fail = 0
    cur_min_lr = min_lr
    cur_max_lr = max_lr

    delta_eps_schedule = [0.01, 0.003, 0.001, 0]
    update_steps = [1, 10, 100, 100]
    update_weight = [2, 1.5, 1.5, 1.5]
    cur_eps_period = 0

    group_gen = EquallySplitGrouping(image_split)

    while num_iter < max_iter:
        top_val, top_idx, _ = vid_model(adv_vid[None, :])
        num_iter += 1

        tentative_directions = directions_generator(adv_vid).cuda()
        group_gen.initialize(tentative_directions)

        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        target_class, rank_transform, sub_num_sample, group_gen, untargeted=False)
        if l is None and g is None:
            logging.info('nes sim fails, try again....')
            continue

        # Rectify tentative perturabtions
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))

        if target_class == top_idx[0][0] and cur_eps <= eps:
            logging.info('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        idx = (top_idx == target_class).nonzero()
        pre_score = top_val[0][idx[0][1]]
        logging.info('cur target prediction: {}'.format(pre_score))
        logging.info('cur eps: {}'.format(cur_eps))

        num_iter += sample_per_draw

        cur_lr = cur_max_lr
        prop_de = delta_eps

        while True:
            num_iter += 1
            proposed_adv_vid = adv_vid.clone()

            assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
            # PGD
            proposed_adv_vid -= cur_lr * rectified_directions
            proposed_eps = max(cur_eps - prop_de, eps)
            bottom_bounded_adv = torch.where((vid - proposed_eps) > proposed_adv_vid, vid - proposed_eps,
                                             proposed_adv_vid)
            bounded_adv = torch.where((vid + proposed_eps) < bottom_bounded_adv, vid + proposed_eps, bottom_bounded_adv)
            clip_frame = torch.clamp(bounded_adv, 0., 1.)
            proposed_adv_vid = clip_frame.clone()

            top_val, top_idx, _ = vid_model(proposed_adv_vid[None, :])
            if target_class in top_idx[0]:
                logging.info('update with delta eps: {}'.format(prop_de))
                if prop_de > 0:
                    cur_max_lr = max_lr
                    cur_min_lr = min_lr
                    explore_succ.clear()
                    reduce_eps_fail = 0
                else:
                    explore_succ.append(True)
                    reduce_eps_fail += 1

                adv_vid = proposed_adv_vid.clone()
                cur_eps = max(cur_eps - prop_de, eps)
                break
            # Adjust the learning rate
            elif cur_lr >= cur_min_lr * 2:
                cur_lr = cur_lr / 2
            else:
                if prop_de == 0:
                    explore_succ.append(False)
                    reduce_eps_fail += 1
                    logging.info('Trying to eval grad again.....')
                    break
                prop_de = 0
                cur_lr = cur_max_lr

        # Adjust delta eps
        if reduce_eps_fail >= update_steps[cur_eps_period]:
            delta_eps = max(delta_eps / update_weight[cur_eps_period], delta_eps_schedule[cur_eps_period])
            logging.info('Success rate of reducing eps is too low. Decrease delta eps to {}'.format(delta_eps))
            if delta_eps <= delta_eps_schedule[cur_eps_period]:
                cur_eps_period += 1
            if delta_eps < 1e-5:
                logging.info('fail to converge at query number {} with eps {}'.format(num_iter, cur_eps))
                return False, cur_eps, adv_vid
            reduce_eps_fail = 0

        # Adjust the max lr and min lr
        if len(explore_succ) == explore_succ.maxlen and cur_min_lr > 1e-7:
            succ_p = np.mean(explore_succ)
            if succ_p < 0.5:
                cur_min_lr /= 2
                cur_max_lr /= 2
                explore_succ.clear()
                logging.info('explore succ rate too low. increase lr scope [{}, {}]'.format(cur_min_lr, cur_max_lr))
        logging.info('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))
    return False, cur_eps, adv_vid


# Input video should be torch.tensor and its shape should be [num_frames, c, w, h]
# The input should be normalized to [0, 1]
def untargeted_video_attack(vid_model, vid, directions_generator, ori_class,
                            rank_transform=False, eps=0.05, max_lr=1e-2, min_lr=1e-3, sample_per_draw=50,
                            max_iter=10000, sigma=1e-5, sub_num_sample=12, image_split=1):
    num_iter = 0
    adv_vid = torch.clamp(vid.clone() + (torch.rand_like(vid) * 2 - 1) * eps, 0., 1.)
    cur_lr = max_lr
    last_p = []
    last_score = []

    group_gen = EquallySplitGrouping(image_split)

    while num_iter < max_iter:
        top_val, top_idx, _ = vid_model(adv_vid[None, :])
        num_iter += 1
        if ori_class != top_idx[0][0]:
            logging.info('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        idx = (top_idx == ori_class).nonzero()
        pre_score = top_val[0][idx[0][1]]
        logging.info('cur target prediction: {}'.format(pre_score))

        last_score.append(pre_score)
        last_score = last_score[-200:]
        if last_score[-1] >= last_score[0] and len(last_score) == 200:
            print('FAIL: No Descent, Stop iteration')
            return False, pre_score.cpu().item(), adv_vid

        # Annealing max learning rate
        last_p.append(pre_score)
        last_p = last_p[-20:]
        if last_p[-1] <= last_p[0] and len(last_p) == 20:
            if cur_lr > min_lr:
                print("[log] Annealing max_lr")
                cur_lr = max(cur_lr / 2., min_lr)
            last_p = []

        tentative_directions = directions_generator(adv_vid).cuda()
        group_gen.initialize(tentative_directions)

        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        ori_class, rank_transform, sub_num_sample, group_gen, untargeted=True)

        if l is None and g is None:
            logging.info('nes sim fails, try again....')
            continue

        # Rectify tentative perturabtions
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))
        num_iter += sample_per_draw

        proposed_adv_vid = adv_vid

        assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
        # PGD
        proposed_adv_vid += cur_lr * rectified_directions
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()

        logging.info('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))
    return False, pre_score.cpu().item(), adv_vid
