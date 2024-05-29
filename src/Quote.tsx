type Quote = {
    quote: string
    author: string
    c: string
}

export default function Quote({quote, author, c} : Quote) {
    return (
        <li id={c}>
            {quote} - {author}
        </li>
    )
}