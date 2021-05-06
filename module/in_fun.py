import pandas as pd
import numpy as np


def self_sorter(ref, contact):
    # 두가지 형태 차, 각각으로 제공
    constant_value = -200
    ref_rssi = ref[1::2][(ref[1::2] != 0)]
    ref_order = np.argsort(ref_rssi)
    ref_idxs = ref[::2][ref_order]

    contact_rssi = contact[1::2][(contact[1::2] != 0)]
    contact_order = np.argsort(contact_rssi)
    contact_idxs = contact[::2][contact_order]
    # sort by ref
    ref_by_ref = ref_rssi[ref_order]
    contact_by_ref = []
    for ref_idx in ref_idxs:
        cur = contact[1::2][(contact[::2] == ref_idx)]
        if len(cur) == 0:
            contact_by_ref.append(constant_value)
        else:
            contact_by_ref.append(cur[0])

    # sort by contact
    contact_by_contact = contact_rssi[contact_order]
    ref_by_contact = []
    for contact_idx in contact_idxs:
        cur = ref[1::2][(ref[::2] == contact_idx)]
        if len(cur) == 0:
            ref_by_contact.append(constant_value)
        else:
            ref_by_contact.append(cur[0])
    srted_list = []
    srted_list.append(ref_by_ref)
    srted_list.append(ref_by_contact)
    srted_list.append(contact_by_ref)
    srted_list.append(contact_by_contact)

    return srted_list


def pad_and_norm(srted_list, data_size=20):
    constant_value = -200
    padded_list = []
    for srted in srted_list:
        if len(srted) > data_size:
            padded = np.array(srted[:data_size])
        else:
            padded = np.pad(
                srted, (data_size - len(srted), 0), constant_values=constant_value
            )
        padded += 100
        padded = padded / 100
        padded_list.append(padded)

    inp_ref = np.concatenate([padded_list[0], padded_list[1]])
    inp_contact = np.concatenate([padded_list[2], padded_list[3]])

    return inp_ref, inp_contact


def test_matcher(a, b):
    srted_list = self_sorter(a, b)
    inp_ref, inp_contact = pad_and_norm(srted_list)
    return np.concatenate([inp_ref, inp_contact])


def train_matcher(a, b):
    # 세가지 형태 (차abs, 차, 각각으로 제공)
    bcast_a, bcast_b = broadcaster(a, b)
    inp_ref_list = []
    inp_contact_list = []
    for a_line, b_line in zip(bcast_a, bcast_b):
        srted_list = self_sorter(a_line, b_line)
        inp_ref, inp_contact = pad_and_norm(srted_list)
        inp_ref_list.append(inp_ref)
        inp_contact_list.append(inp_contact)
    inp_ref_arr = np.array(inp_ref_list)
    inp_contact_arr = np.array(inp_contact_list)
    return  np.concatenate([inp_ref_arr, inp_contact_arr], axis=1)


def matcher_old(ref, contact):
    # 세가지 형태 (차abs, 차, 각각으로 제공)
    bcast_ref, bcast_b = broadcaster(ref, contact)
    srt_ref, srt_contact = sorter(bcast_ref, bcast_b)
    inp_diff = srt_ref - srt_contact
    inp_each = np.concatenate([srt_ref, srt_contact], axis=1)

    return inp_diff, inp_each


def parser(data, beacon_list, data_length=23, beacon_length=10):
    total_beacon = len(beacon_list)
    parsed = np.zeros([np.shape(data)[0], data_length + total_beacon]) - 200
    parsed[:, :data_length] = data[:, :data_length]

    for beacon in np.arange(beacon_length):
        col_idx = data_length + beacon * 4
        having_beacon = data[data[:, col_idx] != 0]
        for idx in range(np.shape(having_beacon)[0]):
            beacon_id = int(having_beacon[idx, col_idx])
            beacon_idx = beacon_list.index(beacon_id)
            beacon_rssi = having_beacon[idx, col_idx + 1]
            parsed[idx, data_length + beacon_idx] = beacon_rssi
    return parsed


def sorter(a, b, data_size=9):
    a_total = []
    b_total = []
    for idx in range(np.shape(a)[0]):
        a_len = 9 - len(a[idx][a[idx] != -1])
        b_len = 9 - len(b[idx][b[idx] != -1])
        a_idx = np.argsort(a[idx])[a_len:]
        b_idx = np.argsort(b[idx])[b_len:]

        a_by_a = a[idx][a_idx]
        a_by_b = a[idx][b_idx]
        a_by_b = a_by_b[a_by_b != -1]
        b_by_a = b[idx][a_idx]
        b_by_a = b_by_a[b_by_a != -1]
        b_by_b = b[idx][b_idx]
        srted_list = []
        srted_list.append(a_by_a)
        srted_list.append(a_by_b)
        srted_list.append(b_by_a)
        srted_list.append(b_by_b)

        padded_list = []
        for srted in srted_list:
            if len(srted) > data_size:
                padded = np.array(srted[:data_size])
            else:
                padded = np.pad(srted, (data_size - len(srted), 0), constant_values=-1)
            padded_list.append(padded)

        a_total.append(np.concatenate([padded_list[0], padded_list[1]]))
        b_total.append(np.concatenate([padded_list[2], padded_list[3]]))

    return np.array(a_total), np.array(b_total)


def sorter_old(a, b):
    a_index = a.argsort(axis=-1)
    b_index = b.argsort(axis=-1)
    a_sortedby_a = np.take_along_axis(a, a_index, axis=-1)
    b_sortedby_a = np.take_along_axis(b, a_index, axis=-1)
    a_sortedby_b = np.take_along_axis(a, b_index, axis=-1)
    b_sortedby_b = np.take_along_axis(b, b_index, axis=-1)
    return np.concatenate((a_sortedby_a, a_sortedby_b), axis=1), np.concatenate(
        (b_sortedby_a, b_sortedby_b), axis=1
    )


def broadcaster(a, b):
    lst_ax = np.shape(a)[-1]
    bcast_a = np.broadcast_to(
        a.reshape([len(a), 1, lst_ax]), (len(a), len(b), lst_ax)
    ).reshape([len(a) * len(b), lst_ax])
    bcast_b = np.broadcast_to(
        b.reshape([1, len(b), lst_ax]), (len(a), len(b), lst_ax)
    ).reshape([len(a) * len(b), lst_ax])

    return bcast_a, bcast_b
