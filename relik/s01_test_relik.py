import argparse
import json
import pdb

from relik.inference.data.objects import RelikOutput

from relik import Relik

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=False, type=str,
                        default='sapienzanlp/relik-entity-linking-large',
                        help='The config file that contains all the parameters')

    # parser.add_argument('--device',
    #                     help='The device to load tensors to, examples: "cpu", "cuda:1", '
    #                          '"cuda"....',
    #                     type=str,
    #                     required=False,
    #                     default='cpu')
    # parser.add_argument('--device2',
    #                     help='The device to do the in operation',
    #                     type=str,
    #                     required=False,
    #                     default='cpu')
    #
    # parser.add_argument('--batch_size',
    #                     help='The size of the batch',
    #                     type=int,
    #                     required=False,
    #                     default=1)

    # parser.add_argument('--batching_type',
    #                     help='The type of method to use for batching',
    #                     type=str,
    #                     required=False,
    #                     default='method1')
    ####
    ####
    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))

    relik = Relik.from_pretrained(args.model_name)
    pdb.set_trace()
    relik_out: RelikOutput = relik('Michael Jordan was one of the best players in the NBA.')
    pdb.set_trace()