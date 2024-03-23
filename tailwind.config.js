module.exports = {
    mode: "all",
    content: [
        "*/src/*.{rs,html,css}",
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
