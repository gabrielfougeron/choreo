from pygments.style import Style
from pygments.token import Comment, Keyword, Name, Number, Operator, \
    Punctuation, String, Token


class PythonVSMintedStyle(Style):

    background_color = '#282C34'

    styles = {
        Token:                  '#ABB2BF',

        Punctuation:            '#46aeef',
        Punctuation.Marker:     '#ABB2BF',

        Keyword:                '#C678DD',
        Keyword.Constant:       '#E5C07B',
        Keyword.Declaration:    '#C678DD',
        Keyword.Namespace:      '#C678DD',
        Keyword.Reserved:       '#C678DD',
        Keyword.Type:           '#E5C07B',

        Name:                   '#ABB2BF',
        Name.Attribute:         '#d19a66',
        Name.Builtin:           '#56b6c2',
        Name.Class:             '#E5C07B',
        Name.Function:          '#61afef',
        Name.Function.Magic:    'bold #56B6C2',
        Name.Other:             '#E06C75',
        Name.Tag:               '#E06C75',
        Name.Decorator:         '#61AFEF',
        Name.Variable.Class:    '#d19a58',

        String:                 '#89c379',

        Number:                 '#d19a58',

        Operator:               '#C678DD',

        Comment:                'italic #7F848E'
    }
