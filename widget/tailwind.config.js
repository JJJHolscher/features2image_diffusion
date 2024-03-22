module.exports = {
    mode: "all",
    content: [
        // include all rust, html and css files in the src directory
        "./src/**/*.{rs,html,css}",
        // include all html files in the output (dist) directory
        "*.html",
        "doc/*.html",
        "web/**/*.html",
        "web/*.html",
    ],
    theme: {
        extend: {
            gridTemplateColumns: {
                'auto': 'repeat(auto-fit, minmax(50px, 1fr))',
            },
            gridAutoRows: {
                '20': '20px',
            }
        },
    },
    plugins: [],
};
