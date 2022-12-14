import time
import os,re
import string
import queue
import encoder
from tqdm import tqdm
import sys

enc = encoder.get_encoder("BART")
try:
    field_empty = enc.pad_token_id
    if not field_empty:
        field_empty = enc.eos_token_id
except:
    field_empty = enc.eos_token_id
eos = enc.eos_token_id
bos = enc.bos_token_id
train_vocab=len(enc)


def join_box(list_in):
    """
    Filters empty fields, combines multiple values into same field
    Args:
        list_in: list of field value pairs

    Returns:
        List of tuples of (field_name, (value1, value2, ...))
    """

    out_list = []
    current_name = ""
    current_value = ""

    for each_item in list_in:
        field_name = each_item.split(":")[0]
        field_value = each_item.split(":")[1]

        if field_name == "":
            continue

        if not field_name[-1].isdigit():
            if field_value != "<none>":
                out_list.append((field_name, field_value))
            continue

        field_name = "_".join(field_name.split("_")[:-1])

        if field_name != current_name:
            if current_name != "":
                # remove none value
                if current_value.strip() != "<none>":
                    out_list.append((current_name, current_value.strip()))
                current_name = ""
                current_value = ""

        current_name = field_name
        current_value += (field_value + " ")

    if current_value.strip() != "<none>":
        out_list.append((current_name, current_value.strip()))

    sorted_by_second = sorted(out_list, key=lambda tup: len(tup[1].split(" ")), reverse=True)

    return out_list, sorted_by_second


def load_dem_map(file_in):
    """
    recursively load nationality map
    Args:
        file_in:

    Returns:

    """
    dem_map = {}
    with open(file_in) as f:
        for line in f:
            line_list = line.strip().lower().split(",")
            if line_list[0] not in dem_map:
                dem_map[line_list[0]] = []
            if line_list[1] not in dem_map[line_list[0]]:
                dem_map[line_list[0]].append(line_list[1])

            if line_list[1] not in dem_map:
                dem_map[line_list[1]] = []
            if line_list[0] not in dem_map[line_list[1]]:
                dem_map[line_list[1]].append(line_list[0])

    final_res_map = {}
    for each_con in dem_map:
        res_con = []
        q = queue.Queue()
        q.put(each_con)

        while not q.empty():
            con = q.get()
            if con in res_con:
                continue

            res_con.append(con)
            if con in dem_map:
                for each_sub in dem_map[con]:
                    q.put(each_sub)

        final_res_map[each_con] = res_con

    return final_res_map


def fuzzy_match_rep(source, substring, field_name):
    # TODO
    """

    Args:
        source:
        substring:
        field_name:

    Returns:

    """

    this_value = substring
    out_summary = source

    this_value_list_raw = this_value.split(" ")
    out_summary_list = out_summary.split(" ")
    # print this_value_list
    # print out_summary_list

    this_value_list = []
    for token in this_value_list_raw:
        if not(token in string.punctuation) \
            and token != "(" \
            and token != ")" \
            and token != "-lsb-" \
            and token != "-rsb-":
            this_value_list.append(token)

    if len(this_value_list) == 0:
        return out_summary

    num_consist = 0
    min_index = len(out_summary_list) + 1
    max_index = -1

    for token in this_value_list:
        if token in out_summary_list:
            num_consist += 1
            this_ind = out_summary_list.index(token)
            if this_ind < min_index:
                min_index = this_ind
            if this_ind > max_index:
                max_index = this_ind

    # print num_consist
    # print min_index
    # print max_index


    if float(num_consist) / len(this_value_list) > 0.4:
        if max_index - min_index <= 2 * len(this_value_list):
            ### regard as match
            to_replace = " ".join(out_summary_list[min_index:max_index+1])
            replace_len = len(to_replace.split(" "))
            if out_summary.startswith(to_replace):
                out_summary = out_summary.replace(to_replace + " ", ("<" + field_name + "> ") * replace_len)
            else:
                out_summary = out_summary.replace(" " + to_replace + " ", " " + ("<" + field_name + "> ") * replace_len)

    return out_summary

def argsort(seq):
    pureseq=[x for x in seq if x !=0]
    pureseqidx=[i for i in range(len(seq)) if seq[i]!=0]
    orderedseqidx=sorted(range(len(pureseq)), key=pureseq.__getitem__)
    for i in range(len(orderedseqidx)):
        seq[pureseqidx[i]] = orderedseqidx[i]+1
    return seq


def gen_content_plan(dem_file, in_summary, in_box, out_plans, out_order):
    '''
    Args:
        dem_file: demonymns file
        in_summary: str, summary file
        in_box: str, box file
        out_plans: content plan path
    '''
    ### load nationality demonyms.csv
    dem_map = load_dem_map(dem_file)
    with open(in_box) as f:
        lines_box = f.readlines()
    with open(in_summary) as f:
        lines_summary = f.readlines()

    out_plans = open(out_plans, "w")
    out_order = open(out_order, "w")
    for box, summary in tqdm(zip(lines_box, lines_summary)):
        box = box.replace("-lrb-", "(")
        box = box.replace("-rrb-", ")")

        box_list = box.strip().split("\t")
        box_out_list, box_field_list = join_box(box_list)

        summary = summary.replace("-lrb-", "(")
        summary = summary.replace("-rrb-", ")")
        out_summary = summary.strip()
        out_name=[]
        out_plan=[0]*len(box_field_list)
        order=0
        #print("Summary is: ", out_summary)
        for (this_name, this_value) in box_field_list:
            if " " + this_value + " " in out_summary:
                idx = out_summary.index(" " + this_value + " ")+1
                out_plan[order]= idx
                out_name.append(this_name)
            elif  out_summary.startswith(this_value + " "):
                idx = out_summary.index(this_value + " ")+1
                out_plan[order]= idx
                out_name.append(this_name)
            elif this_value in dem_map:
                for trans_value in dem_map[this_value]:
                    if trans_value in out_summary:
                        idx = out_summary.index(trans_value)+1
                        out_plan[order]= idx
                        out_name.append(this_name)
                        break
            #else:
                #print(this_value," not in gold summary!")
            order+=1
        plan = argsort(out_plan)
        tmp = [out_name[x-1] for x in plan if x!=0]
        out_order.write(" ".join(list(map(str, plan ))) + "\n")
        out_plans.write(" ".join(tmp) + "\n")
    out_order.close()
    out_plans.close()


def gen_mask_field_pos(dem_file, in_summary, in_box, out_field, out_pos, out_rpos, out_plans, out_orders, key2id, keyid2wordlist):
    #TODO: ?????????mask?????????box???mask???????????????content selector
    """
    Mask out the values in the table content by whether they occur in summary
    Args:
        dem_file: demonymns file
        in_summary: str, summary file
        in_box: str, box file
        out_field: masked summary
        out_pos: summary with field position values
        out_rpos: summary with reversed field position values

    Returns:
        None
    """

    ### load nationality demonyms.csv
    dem_map = load_dem_map(dem_file)

    with open(in_box) as f:
        lines_box = f.readlines()

    with open(in_summary) as f:
        lines_summary = f.readlines()

    out_s = open(out_field, "w")
    out_p = open(out_pos, "w")
    out_rp = open(out_rpos, "w")
    out_plans = open(out_plans, "w")
    out_orders = open(out_orders, "w")

    max_table_len=0
        
    a,b=in_box.split("/original_data/")
    split=b.split(".box")[0]
    fn=os.path.join(a,"processed_data",split,b+".val")
    enc_key=os.path.join(a,"processed_data",split,b+".iskey")
    out_enckey = open(enc_key, "w")
    

    with open(fn, "r") as h:
        lines_boxval=h.readlines()
    assert len(lines_boxval)==len(lines_box)
    for boxval, box, summary in tqdm(zip(lines_boxval, lines_box, lines_summary)):
        boxval = boxval.replace("-lrb-", "(")
        boxval = boxval.replace("-rrb-", ")")
        boxval = boxval.strip()
        boxval_idx = [0]*len(boxval.split())
        box = box.replace("-lrb-", "(")
        box = box.replace("-rrb-", ")")

        box_list = box.strip().split("\t")
        box_out_list, box_field_list = join_box(box_list)


        summary = summary.replace("-lrb-", "(")
        summary = summary.replace("-rrb-", ")")

        
        tem_summary = summary.strip()
        out_summary = summary.strip()
        tem_summary_list = tem_summary.split(" ")
        
        out_name = []
        out_plan = []
        out_order = [0]*len(box_field_list)
        max_table_len = max(len(box_field_list), max_table_len)
        out_pos, out_rpos, out_field = [], [], []
        out_pos_bpe, out_rpos_bpe, out_field_bpe = [], [], []

        out_bpe = enc.encode(summary.strip())
        out_bpe_len = len(out_bpe)
        #TODO:??????out_bpe???boxval??????????????????value
        
        #initialize sentence
        for ind in range(out_bpe_len):
            out_pos_bpe.append(0)
            out_rpos_bpe.append(0)

        for ind in range(out_bpe_len):
            out_field_bpe.append('#')

        for ind in range(len(tem_summary_list)):
            out_pos.append(0)
            out_rpos.append(0)

        for ind in range(len(tem_summary_list)):
            out_field.append('#')

        
        for (this_name, this_value) in box_field_list:
            out_name.append(this_name)
            this_value_dict = {}
            this_pos_bpe_dict = {}
            prev = 1
            for ind, each_token in enumerate(this_value.split(" ")):
                # if each_token not in this_value_dict:
                this_value_dict[each_token] = ind + 1

                if this_name != "name":
                    each_token = " " + each_token
                else:
                    if ind != 0:
                        each_token = " " + each_token

                bpe_tokens = enc.encode(each_token)

                # (start ind, len)
                this_pos_bpe_dict[ind + 1] = (prev, len(bpe_tokens))
                prev += len(bpe_tokens)

            if this_name == "name":
                bpe_value = this_value
            else:
                bpe_value = " " + this_value
            bpe_tokens = enc.encode(bpe_value)

            this_value_bpe_len = len(bpe_tokens)
            this_value_list_len = len(this_value.split(" "))

            if " " + this_value + " " in out_summary:
                out_summary = out_summary.replace(" " + this_value + " ", " " + ("<" + this_name + "> ") * this_value_list_len)
                boxval = boxval.replace(" ".join(enc.tokenize(" "+this_value))+" ", ("<spec> ") * len(enc.tokenize(" "+this_value)))
                boxval = boxval.replace(" ".join(enc.tokenize(this_value))+" ", ("<spec> ") * len(enc.tokenize(this_value)))

            # name
            elif out_summary.startswith(this_value + " "):
                out_summary = out_summary.replace(this_value + " ", ("<" + this_name + "> ") * this_value_list_len)
                boxval = boxval.replace(" ".join(enc.tokenize(this_value))+" ", ("<spec> ") * len(enc.tokenize(this_value)))

            # nationality
            elif this_value in dem_map:
                this_value_list = dem_map[this_value]
                for this_value in this_value_list:
                    this_value_list_len = len(this_value.split(" "))
                    if " " + this_value + " " in out_summary:

                        out_summary = out_summary.replace(" " + this_value + " ", " " + ("<" + this_name + "> ") * this_value_list_len)
                        boxval = boxval.replace(" ".join(enc.tokenize(" "+this_value))+" ", ("<spec> ") * len(enc.tokenize(" "+this_value)))
                        boxval = boxval.replace(" ".join(enc.tokenize(this_value))+" ", ("<spec> ") * len(enc.tokenize(this_value)))
            else:
                # seperate nationality
                is_dem_match = 0
                this_value_list = re.split(r" , | \( | \) | `` | '' | \)|\( |`` | ''", this_value)
                if len(this_value_list) > 1:
                    for each_con in this_value_list:
                        if each_con and " " + each_con + " " in out_summary:
                            each_con_len = len(each_con.split(" "))
                            out_summary = out_summary.replace(" " + each_con + " ", " " + ("<" + this_name + "> ") * each_con_len)
                            boxval = boxval.replace(" ".join(enc.tokenize(" "+each_con)) + " ", ("<spec> ") * len(enc.tokenize(" "+each_con)))
                            break
                        elif each_con and each_con + " " in out_summary:
                            each_con_len = len(each_con.split(" "))
                            out_summary = out_summary.replace(each_con + " ", ("<" + this_name + "> ") * each_con_len)
                            boxval = boxval.replace(" ".join(enc.tokenize(each_con)) + " ", ("<spec> ") * len(enc.tokenize(each_con)))
                            break
                        if each_con in dem_map:
                            this_con_list = dem_map[each_con]
                            for this_con in this_con_list:
                                if " " + this_con + " " in out_summary:
                                    this_con_len = len(this_con.split(" "))
                                    this_con_len = len(this_con.split(" "))
                                    out_summary = out_summary.replace(" " + this_con + " ", " " + ("<" + this_name + "> ") * this_con_len)
                                    boxval = boxval.replace(" ".join(enc.tokenize(" " + this_con)) + " ", ("<spec> ") * len(enc.tokenize(" "+this_con)))
                                    is_dem_match = 1
                                    break

                if is_dem_match:
                    continue
                # do not fuzzy match
                #out_summary = fuzzy_match_rep(out_summary, this_value, this_name)
                #boxval = fuzzy_match_rep(boxval, this_value, this_name)

            assert len(out_summary.split(" ")) == len(tem_summary_list)
            assert len(boxval.split(" ")) == len(boxval_idx)
            
            for ind, token in enumerate(out_summary.split(" ")):
                if token == "<" + this_name + ">":
                    out_field[ind] = this_name
                    ori_token = tem_summary_list[ind]
                    if ori_token in this_value_dict:
                        out_pos[ind] = this_value_dict[ori_token]
                        out_rpos[ind] = this_value_list_len - (out_pos[ind] - 1)

                    # convert to bpe
                    ori_token_bpe = ori_token
                    if ind != 0:
                        ori_token_bpe = " " + ori_token

                    if ind > 0:
                        past = tem_summary_list[:ind]
                        past = " ".join(past)
                        bpe_past = enc.encode(past)
                        past_len = len(bpe_past)

                    else:
                        past_len = 0

                    bpe_tokens = enc.encode(ori_token_bpe)
                    for it in range(len(bpe_tokens)):
                        out_field_bpe[past_len + it] = this_name

                    if ori_token in this_value_dict:
                        bpe_pos_start, bpe_pos_len = this_pos_bpe_dict[out_pos[ind]]
                        for it in range(bpe_pos_len):
                            start = bpe_pos_start + it
                            end = this_value_bpe_len - (start - 1)
                            if start > 30:
                                start = 30
                            if end > 30:
                                end = 30
                            if past_len + it >= len(out_pos_bpe):
                                this_id = past_len
                            else:
                                this_id = past_len + it
                            out_pos_bpe[this_id] = start
                            out_rpos_bpe[this_id] = end
        for ind, token in enumerate(boxval.split(" ")):
                if token=="<spec>":
                    boxval_idx[ind]=1
                else:
                    boxval_idx[ind]=0
        bpe_tokens = enc.encode(summary.strip())

        assert len(out_summary.split(" ")) == len(tem_summary_list)

        assert len(out_field) == len(tem_summary_list)
        assert len(tem_summary_list) == len(out_pos)
        assert len(tem_summary_list) == len(out_rpos)

        assert len(out_field_bpe) == len(bpe_tokens)
        assert len(out_pos_bpe) == len(bpe_tokens)
        assert len(out_rpos_bpe) == len(bpe_tokens)
        
        #generate content plan
        prev=None
        for i in range(len(out_field_bpe)):
            if out_field_bpe[i]!='#' and out_field_bpe[i]!=prev:
                prev = out_field_bpe[i]
                out_plan.append(out_field_bpe[i])
        for i in range(len(out_order)):
            if out_name[i] in out_plan:
                out_order[i] = out_plan.index(out_name[i])+1

        #end generation

        
        out_orders.write(" ".join(list(map(str, out_order ))) + "\n")
        tmp = [key2id[key] if key in key2id else 0 for key in out_plan]
        out_plan_id=[]
        for idx,x in enumerate(tmp):
            field_token_id = []
            for token in keyid2wordlist[x]:
                if token!=field_empty:
                    field_token_id.append(str(token))
            out_plan_id.extend(field_token_id)
            if idx!=len(tmp)-1:
                out_plan_id.extend(list(map(str, enc.encode(","))))
        out_plan_id.extend(list(map(str, enc.encode("|"))))#??????????????????hard prompt??????????????????key1,key2,key3|context
        out_plans.write(" ".join(out_plan_id) + "\n")
    
        out_s.write(" ".join(out_field_bpe) + "\n")
        out_p.write(" ".join([str(tmp) for tmp in out_pos_bpe]) + "\n")
        out_rp.write(" ".join([str(tmp) for tmp in out_rpos_bpe]) + "\n")
        out_enckey.write(" ".join([str(tmp) for tmp in boxval_idx]) + "\n")

    print("Maximum table length: ", max_table_len)
    out_orders.close()
    out_plans.close()
    out_s.close()
    out_p.close()
    out_rp.close()
    out_enckey.close()



def gen_context(subdir):
    """
    Process box data to use as input to GPT
    Args:
        subdir: str, root path

    Returns:
        None
    """
    boxes = []
    context = []
    contextid=[]
    for split in ["train", "valid", "test"]:
        boxes.append(os.path.join(subdir, "original_data", split + ".box"))
        context.append(os.path.join(subdir, "processed_data", split, split + ".context"))
        contextid.append(os.path.join(subdir, "processed_data", split, split + ".context.id"))

    avg_len = 0
    num = 0
    for ind, fboxes in enumerate(boxes):
        box = open(fboxes, "r").read().strip().split('\n')
        context_out = open(context[ind], "w")
        contextid_out = open(contextid[ind], "w")
        for ib in box:
            ib = ib.replace("-lrb-", "(")
            ib = ib.replace("-rrb-", ")")
            item = ib.split('\t')
            box_out_list, _ = join_box(item)

            write_line, write_line2 = [],[]
            for (this_name, this_value) in box_out_list:

                if '<none>' in this_value:
                    continue

                to_write = ""
                # if this_name == "name":
                #     # for humans
                #     if domain == "humans":
                #         to_write = "name is " + this_value + " ,"
                #     # to_write = "name ,"

                #     # for books
                #     if domain == "books":
                #         to_write = "title is " + this_value + " ,"

                #     # for songs
                #     if domain == "songs":
                #         to_write = "song name is " + this_value + " ,"
                
                # else:
                #     write_value = " " + this_value
                #     write_name = " " + this_name.replace("_", " ")
                #     to_write = write_name + " is" + write_value + " ,"
                write_name = this_name.replace("_", " ")
                to_write =  write_name + " is " + this_value + " ; "
                
                tokens = enc.tokenize(to_write)
                tokens_id = enc.convert_tokens_to_ids(tokens)

                write_line.extend(tokens)
                write_line2.extend(tokens_id)
            assert len(write_line2)==len(write_line)
            avg_len += len(write_line)
            num += 1
            context_out.write(" ".join([str(tmp) for tmp in write_line]) + "\n")
            contextid_out.write(" ".join([str(tmp) for tmp in write_line2]) + "\n")

        context_out.close()
        contextid_out.close()
        print(float(avg_len) / num)


def split_infobox(subdir):
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the begining of a field
    """
    bwfile = []
    bffile = []
    bpfile = []
    blfile = []
    boxes = []
    for split in ['train', 'test', 'valid']:
        bwfile.append(os.path.join(subdir, 'processed_data', split, split + '.box.val'))
        bffile.append(os.path.join(subdir, 'processed_data', split, split + '.box.lab'))
        bpfile.append(os.path.join(subdir, 'processed_data', split, split + '.box.pos'))
        blfile.append(os.path.join(subdir, 'processed_data', split, split + '.box.parent'))
        boxes.append(os.path.join(subdir, 'original_data', split + '.box'))


    mixb_word, mixb_label, mixb_pos, mixb_parent = [], [], [], []
    for fboxes in boxes:
        box = open(fboxes, "r").read().strip().split('\n')
        box_word, box_label, box_pos, box_parent  = [], [], [], []
        for ib in box:

            ib = ib.replace("-lrb-", "(")
            ib = ib.replace("-rrb-", ")")

            box_single_word, box_single_label, box_single_pos, box_single_parent = [], [], [], ""
            item = ib.split('\t')

            box_out_list, _ = join_box(item)

            for (this_name, this_value) in box_out_list:

                if '<none>' in this_value or not this_name or not this_value:
                    continue
                this_name=this_name.strip("\t|\|\|\|")
                this_value=this_value.strip("\t|\|\|\|")
                box_single_parent += this_name + "|||" + this_value + "\t"
                if this_name != "name":
                    this_value = " " + this_value

                tokens_original = enc.tokenize(this_value)
                for ind, each_token in enumerate(tokens_original):
                    box_single_word.append(each_token)
                    box_single_label.append(this_name)
                    box_single_pos.append(ind + 1  if ind + 1<=30 else 30)

            box_word.append(box_single_word)
            box_label.append(box_single_label)
            box_pos.append(box_single_pos)
            box_parent.append(box_single_parent)
        mixb_parent.append(box_parent)
        mixb_word.append(box_word)
        mixb_label.append(box_label)
        mixb_pos.append(box_pos)
    for k, m in enumerate(mixb_word):
        with open(bwfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_label):
        with open(bffile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_pos):
        with open(bpfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_parent):
        with open(blfile[k], "w+") as h:
            for items in m:
                h.write(items.strip('\t') + "\n")


def reverse_pos(subdir):
    """
    get the position counted from the end of a field
    Args:
        subdir: str, root directory

    Returns:
        None
    """
    bpfile = []
    bwfile = []
    for split in ['train', 'test', 'valid']:
        bpfile.append(os.path.join(subdir, 'processed_data', split, split + '.box.pos'))
        bwfile.append(os.path.join(subdir, 'processed_data', split, split + '.box.rpos'))

    for k, pos in enumerate(bpfile):
        box = open(pos, "r").read().strip().split('\n')
        reverse_pos = []
        for bb in box:
            pos = bb.split()
            tmp_pos = []
            single_pos = []
            for p in pos:
                if int(p) == 1 and len(tmp_pos) != 0:
                    single_pos.extend(tmp_pos[::-1])
                    tmp_pos = []
                tmp_pos.append(p)
            single_pos.extend(tmp_pos[::-1])
            reverse_pos.append(single_pos)
        with open(bwfile[k], 'w+') as bw:
            for item in reverse_pos:
                bw.write(" ".join(item) + '\n')


def check_generated_box(subdir):
    """
    Check len of input data matches
    Args:
        subdir: str, root path

    Returns:
        None
    """
    ftrain = []
    ftest = []
    fvalid = []
    for fp in [".box.val", ".box.lab", ".box.pos", ".box.rpos"]:
        ftrain.append(os.path.join(subdir, 'processed_data', "train", "train" + fp))
        ftest.append(os.path.join(subdir, 'processed_data', "test", "test" + fp))
        fvalid.append(os.path.join(subdir, 'processed_data', "valid", "valid" + fp))

    for case in [ftrain, ftest, fvalid]:
        vals = open(case[0], 'r').read().strip().split('\n')
        labs = open(case[1], 'r').read().strip().split('\n')
        poses = open(case[2], 'r').read().strip().split('\n')
        rposes = open(case[3], 'r').read().strip().split('\n')
        assert len(vals) == len(labs)
        assert len(poses) == len(labs)
        assert len(rposes) == len(poses)
        for val, lab, pos, rpos in zip(vals, labs, poses, rposes):
            vval = val.strip().split(' ')
            llab = lab.strip().split(' ')
            ppos = pos.strip().split(' ')
            rrpos = rpos.strip().split(' ')
            if len(vval) != len(llab) or len(llab) != len(ppos) or len(ppos) != len(rrpos):
                print(case)
                print(val)
                print(len(vval))
                print(len(llab))
                print(len(ppos))
                print(len(rrpos))
            assert len(vval) == len(llab)
            assert len(llab) == len(ppos)
            assert len(ppos) == len(rrpos)


def split_summary_for_rouge(subdir):
    """
    Write each valid and test each example into a different file
    Args:
        domain: str, root folder

    Returns:

    """
    bpfile = []
    bwfile = []
    for split in ["valid", "test"]:
        bpfile.append(os.path.join(subdir, 'original_data', split + '.summary'))
        bwfile.append(os.path.join(subdir, 'processed_data', split, split + '_split_for_rouge'))

    for i, fi in enumerate(bpfile):
        fread = open(fi, 'r')
        k = 0
        for line in fread:
            with open(bwfile[i] + '/gold_summary_' + str(k), 'w') as sw:
                sw.write(line.strip() + '\n')
            k += 1
        fread.close()


def table2id(subdir, merge_field_vocab, dem_file):
    """
        Main pre-processing script that creates masked summaries, writes out tokenized field, value,
        summary and masked summary
    Args:
        domain: str, root path

    Returns:
        None
    """
    fvals = []
    flabs = []
    fparent = []
    fsums = []
    fvals2id = []
    fkey2id = []
    fval2id = []
    flabs2id = []
    fsums2id = []
    f_local_vocab = []
    f_decoder_field = []
    f_plan = []
    f_order=[]
    f_decoder_field_id = []
    f_decoder_pos = []
    f_decoder_rpos = []
    boxes = []
    for split in ["train", "test", "valid"]:
        fvals.append(os.path.join(subdir, 'processed_data', split, split + '.box.val'))
        flabs.append(os.path.join(subdir, 'processed_data', split, split + '.box.lab'))
        fsums.append(os.path.join(subdir, 'original_data', split + '.summary'))
        fparent.append(os.path.join(subdir, 'processed_data', split, split + '.box.parent'))

        fvals2id.append(os.path.join(subdir, 'processed_data', split, split + '.box.val.id'))
        flabs2id.append(os.path.join(subdir, 'processed_data', split, split + '.box.lab.id'))
        fsums2id.append(os.path.join(subdir, 'processed_data', split, split + '.summary.id'))
        fkey2id.append(os.path.join(subdir, 'processed_data', split, split + '.key.id'))
        fval2id.append(os.path.join(subdir, 'processed_data', split, split + '.value.id'))

        f_local_vocab.append(os.path.join(subdir, 'processed_data', split, split + '_local_oov.txt'))
        f_order.append(os.path.join(subdir, 'processed_data', split, split + '_content_plan_id.txt'))
        f_plan.append(os.path.join(subdir, 'processed_data', split, split + '_content_plan.txt'))
        f_decoder_field.append(os.path.join(subdir, 'processed_data', split, split + '_summary_field.txt'))
        f_decoder_field_id.append(os.path.join(subdir, 'processed_data', split, split + '_summary_field_id.txt'))

        f_decoder_pos.append(os.path.join(subdir, 'processed_data', split, split + '_summary_pos.txt'))
        f_decoder_rpos.append(os.path.join(subdir, 'processed_data', split, split + '_summary_rpos.txt'))

        boxes.append(os.path.join(subdir, 'original_data', split + '.box'))

    # write field to word mapping
    key_map = dict()
    key_map['#'] = 0
    cnt = 1
    with open(merge_field_vocab, "r") as v:
        for line in v:
            key = line.strip().split()[0]
            key_map[key] = cnt
            cnt += 1
    key2id = key_map
    id2key = {value: key for key, value in key_map.items()}
    print(len(key_map))

    # add for field id to word group mapping
    keyid2wordlist = dict()
    for i in range(0, len(id2key)):
        if i == 0:
            bpe_in = id2key[i].replace("_", " ")
        else:
            bpe_in = " " + id2key[i].replace("_", " ")
        bpe_tokens = enc.encode(bpe_in)
        keyid2wordlist[i] = bpe_tokens

        if len(keyid2wordlist[i]) > 3:
            keyid2wordlist[i] = keyid2wordlist[i][:3]
        else:
            extended = 3 - len(keyid2wordlist[i])
            keyid2wordlist[i] += ([field_empty] * extended)

    field2word_file = os.path.join(subdir, "processed_data", "field2word.txt")
    with open(field2word_file, "w") as f:
        for each_id in keyid2wordlist:
            f.write(str(each_id) + "\t" + " ".join([str(tmp) for tmp in keyid2wordlist[each_id]]) + "\n")

    # write out field data tokens
    for k, ff in enumerate(flabs):
        fi = open(ff, 'r')
        fo = open(flabs2id[k], 'w')
        for line in fi:
            items = line.strip().split()
            # print (items)
            res_items = []
            for key in items:
                if key in key2id:
                    res_items.append(str(key2id[key]))
                else:
                    res_items.append("0")

            fo.write(" ".join(res_items) + '\n')
        fi.close()
        fo.close()

    # gen field masked summary
    for k, (fs, fb) in enumerate(zip(fsums, boxes)):
        #gen_content_plan(dem_file, fs, fb, f_plan[k], f_order[k])
        gen_mask_field_pos(dem_file, fs, fb, f_decoder_field[k], f_decoder_pos[k], f_decoder_rpos[k], f_plan[k], f_order[k], key2id, keyid2wordlist)

    # write out masked summary tokens
    for k, ff in enumerate(f_decoder_field):
        fi = open(ff, 'r')
        fo = open(f_decoder_field_id[k], 'w')
        for line in fi:
            items = line.strip().split()
            res_items = []
            for key in items:
                if key in key2id:
                    res_items.append(str(key2id[key]))
                else:
                    res_items.append("0")
            fo.write(" ".join(res_items) + '\n')
        fi.close()
        fo.close()

    for j , fpar in enumerate(fparent):
        fp = open(fpar)
        fko = open(fkey2id[j], 'w')
        fvo = open(fval2id[j], 'w')
        lines_parent = fp.readlines()
        for line in lines_parent:
            line_ko=[]
            line_vo=[]
            pair_lst = line.strip().split('\t')
            for pair in pair_lst:
                key1, value1 = pair.split("|||")
                line_ko.append(enc.encode(key1.replace('_', ' ')))
                line_vo.append(enc.encode(value1))
            fko.write("\t".join([" ".join(list(map(str, ko))) for ko in line_ko])+'\n')
            fvo.write("\t".join([" ".join(list(map(str, vo))) for vo in line_vo])+'\n')
        fko.close()
        fvo.close()

    # write out summary, value tokens
    for k, (fs, fv) in enumerate(zip(fsums, fvals)):
        fsum = open(fs)
        fsumo = open(fsums2id[k], 'w')

        fval = open(fv)
        fvalo = open(fvals2id[k], 'w')

        lines_sum = fsum.readlines()
        lines_val = fval.readlines()

        for line_sum, line_val in zip(lines_sum, lines_val):
            line_val_list = line_val.strip().split()
            res_val_list = [str(x) for x in enc.convert_tokens_to_ids(line_val_list)]

            # res_val_list = [str(enc.encoder[bpe_token]) for bpe_token in line_val_list]
            fvalo.write(" ".join(res_val_list) + "\n")
            line_sum = line_sum.strip()
            line_sum = line_sum.replace("-lrb-", "(")
            line_sum = line_sum.replace("-rrb-", ")")
            res_sum_list = enc.encode(line_sum)
            fsumo.write(" ".join([str(tmp) for tmp in res_sum_list]) + "\n")

        fsumo.close()
        fvalo.close()



def preprocess(subdir, merge_field_vocab, dem_file):
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field. 
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    print("extracting token, field type and position info from original data ...")
    time_start = time.time()

    split_infobox(subdir)
    reverse_pos(subdir)

    duration = time.time() - time_start
    print("extract finished in %.3f seconds" % float(duration))

    print("spliting test and valid summaries for ROUGE evaluation ...")
    time_start = time.time()
    split_summary_for_rouge(subdir)

    duration = time.time() - time_start
    print("split finished in %.3f seconds" % float(duration))

    print("turning words and field types to ids ...")
    time_start = time.time()
    table2id(subdir, merge_field_vocab, dem_file)
    duration = time.time() - time_start
    print("idlization finished in %.3f seconds" % float(duration))

    print("generate prefix table")
    gen_context(subdir)


def make_dirs(subdir):
    """
    Make directoies
    Args:
        subdir: Root directory

    Returns:
        None
    """
    os.mkdir(os.path.join(subdir, "processed_data"))
    os.mkdir(os.path.join(subdir, "processed_data", "train"))
    os.mkdir(os.path.join(subdir, "processed_data", "test"))
    os.mkdir(os.path.join(subdir, "processed_data", "valid"))
    os.mkdir(os.path.join(subdir, "processed_data", "test", "test_split_for_rouge"))
    os.mkdir(os.path.join(subdir, "processed_data", "valid", "valid_split_for_rouge"))


if __name__ == '__main__':

    root_path = sys.argv[1]
    domain = sys.argv[2]
    subdir = os.path.join(root_path, domain)
    dem_file = os.path.join(root_path, "demonyms.csv")
    merge_field_vocab = os.path.join(root_path, "human_books_songs_films_field_vocab.txt")
    make_dirs(subdir)
    preprocess(subdir, merge_field_vocab, dem_file)
    check_generated_box(subdir)
    print("check done")
